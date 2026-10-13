'use strict';
const $ = id => document.getElementById(id);
const esc = s => String(s ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
let user = null, projects = [];

async function api(url, body, method) {
  const opt = body === undefined ? {} : {method: method || 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body)};
  let r; try { r = await fetch(url, opt); } catch { throw Error('Network error. Is the server running?'); }
  if (r.status === 401) { location.href = '/'; throw Error('Please log in.'); }
  let d = {}; try { d = await r.json(); } catch {}
  if (!r.ok) throw Error(d.error || 'Request failed (' + r.status + ')');
  return d;
}
// Run an action: loading spinner, error box.
async function run(outId, btnId, fn) {
  const out = $(outId), btn = btnId && $(btnId);
  out.innerHTML = '<span class="spin"></span>Working…'; if (btn) btn.disabled = true;
  try { out.innerHTML = await fn() || ''; } catch (e) { out.innerHTML = '<div class="err">' + esc(e.message) + '</div>'; }
  if (btn) btn.disabled = false;
}
const empty = m => '<div class="empty">' + esc(m) + '</div>';
const table = (heads, rows) => rows.length ? '<div class="table-responsive"><table><thead><tr>' + heads.map(h => '<th>' + h + '</th>').join('') +
  '</tr></thead><tbody>' + rows.map(r => '<tr>' + r.map(c => '<td>' + c + '</td>').join('') + '</tr>').join('') + '</tbody></table></div>' : '';
const need = (id, label) => { const v = $(id).value.trim(); if (!v) throw Error(label + ' is required.'); return v; };
const chips = a => a.map(x => '<span class="chip">' + esc(x) + '</span>').join('') || '<span class="muted">none</span>';

const rankTable = (res, extra) => res.length ? table(['Rank', 'Title', 'Score'], res.map(r => [r.position, esc(r.title || r.passage), r.score])) : empty('No matching results.');
const NAMES = {tfidf: 'TF-IDF + cosine (0–1, higher is better)', lm: 'Language model log-likelihood (closer to 0 is better)', dfr: 'DFR I(n<sub>e</sub>)B2 (higher is better)'};

const tabs = [['search','Search & Ranking'],['passage','Passage Retrieval'],['rec','Recommendations'],['pr','PageRank'],['eval','Evaluation'],['web','Web Crawler'],['ie','Extraction & Semantic'],['par','Parallel / MapReduce']];
function show(id) {
  document.querySelectorAll('.panel').forEach(p => p.classList.toggle('active', p.id === 'p-' + id));
  document.querySelectorAll('#side button').forEach(b => b.classList.toggle('active', b.dataset.t === id));
  history.replaceState(null, '', '#' + id);
  if (id === 'rec') loadCF(); if (id === 'eval') loadJ();
}
async function logout() { await fetch('/api/auth/logout', {method: 'POST'}); location.href = '/'; }

// ---- Search
$('s-go').onclick = () => run('s-out', 's-go', async () => {
  const q = need('s-q', 'Query'), m = $('s-m').value;
  const d = await api('/api/ir/search', {query: q, method: m});
  if (d.message) return empty(d.message);
  if (m === 'all') return '<div class="cols">' + Object.keys(d.comparison).map(k => '<div><h3>' + NAMES[k] + '</h3>' + rankTable(d.comparison[k]) + '</div>').join('') + '</div>';
  return '<p class="muted">' + NAMES[m] + '</p>' + rankTable(d.results);
});
$('a-go').onclick = () => run('a-out', 'a-go', async () => {
  const d = await api('/api/ir/automaton', {terms: need('a-t', 'Terms'), text: need('a-x', 'Text')});
  return '<div class="info">DFA with ' + d.states + ' states; ' + d.matches.length + ' match(es).</div>' +
    (d.matches.length ? table(['Term', 'Start', 'End'], d.matches.map(m => [esc(m.term), m.start, m.end])) : empty('No term found in the text.')) +
    '<p class="muted mono" style="margin-top:8px">Trace (char→state): ' + d.trace.map(t => esc(t.char) + '→' + t.state).join(' ') + (d.trace_truncated ? ' …' : '') + '</p>';
});

// ---- Passages
$('ps-go').onclick = () => run('ps-out', 'ps-go', async () => {
  const d = await api('/api/ir/passages', {query: need('ps-q', 'Query'), text: need('ps-t', 'Text'), method: $('ps-m').value});
  return rankTable(d.results);
});

// ---- Recommenders
$('r-go').onclick = () => run('r-out', 'r-go', async () => {
  const d = await api('/api/ir/recommendations', {query: need('r-q', 'Query')});
  return d.recommendations.length ? table(['#', 'Title', 'Similarity'], d.recommendations.map((r, i) => [i + 1, esc(r.title), r.similarity_percent + '%'])) : empty('No similar projects found.');
});
$('k-go').onclick = () => run('k-out', 'k-go', async () => {
  const d = await api('/api/ir/recommend/knowledge', {status: $('k-s').value, include: $('k-i').value, exclude: $('k-e').value, min_words: $('k-w').value});
  return d.results.length ? table(['Title', 'Status', 'Matched criteria'], d.results.map(r => [esc(r.title), esc(r.status), esc(r.reasons.join('; '))])) : empty('No project satisfies these requirements.');
});
async function loadCF() {
  if (user.role !== 'student') { $('cf-rate').innerHTML = empty('Collaborative filtering is available to student accounts (ratings come from students).'); $('cf-go').style.display = 'none'; return; }
  const others = projects;
  $('cf-rate').innerHTML = others.length ? table(['Title', 'Your rating'], others.map(p => [esc(p.title),
    '<select data-p="' + p.id + '" class="rate"><option value="">–</option>' + [1,2,3,4,5].map(n => '<option>' + n + '</option>').join('') + '</select>'])) : empty('No projects to rate yet.');
  document.querySelectorAll('.rate').forEach(s => s.onchange = async () => {
    if (!s.value) return;
    try { await api('/api/ir/ratings', {project_id: +s.dataset.p, rating: +s.value}); s.style.borderColor = '#10B981'; }
    catch (e) { s.value = ''; $('cf-out').innerHTML = '<div class="err">' + esc(e.message) + '</div>'; }
  });
}
$('cf-go').onclick = () => run('cf-out', 'cf-go', async () => {
  const d = await api('/api/ir/recommend/collaborative');
  return (d.message ? '<div class="info">' + esc(d.message) + '</div>' : '') +
    (d.recommendations.length ? table(['Title', 'Predicted rating'], d.recommendations.map(r => [esc(r.title), r.predicted_rating])) : '');
});

// ---- PageRank
$('pr-go').onclick = () => run('pr-out', 'pr-go', async () => {
  const d = await api('/api/ir/pagerank', {threshold: parseFloat($('pr-t').value)});
  if (!d.nodes) return empty('No projects in the database yet.');
  return '<div class="info">' + d.nodes + ' nodes, ' + d.edges + ' edges' + (d.edges ? '' : ' — no pairs reach the threshold, so all scores are equal.') + '</div>' +
    table(['Rank', 'Title', 'PageRank', 'Links'], d.results.map(r => [r.position, esc(r.title), r.pagerank, r.links]));
});

// ---- Evaluation
$('e-go').onclick = () => run('e-out', 'e-go', async () => {
  const d = await api('/api/ir/evaluate', {k: +$('e-k').value || 3});
  if (!d.evaluations.length) return empty(d.message || 'No evaluation possible.');
  return '<p class="muted">' + d.queries + ' labelled query(ies).</p>' + table(['Method', 'Mean P@K', 'Mean R@K', 'MAP'],
    d.evaluations.map(e => [e.method.toUpperCase(), e.mean_precision_at_k, e.mean_recall_at_k, e.MAP]));
});
async function loadJ() {
  const isF = user.role === 'faculty';
  $('j-form').style.display = isF ? '' : 'none';
  if (isF) $('j-list').innerHTML = projects.length ? projects.map(p => '<label style="font-weight:400"><input type="checkbox" style="width:auto" value="' + p.id + '"> ' + esc(p.title) + '</label>').join('') : empty('No projects yet.');
  else $('j-hint').textContent = '';
  await run('j-out', null, async () => {
    const d = await api('/api/ir/judgments');
    return d.judgments.length ? table(['Query', 'Relevant project', ''], d.judgments.map(j => [esc(j.query), esc(j.title), isF ? '<button class="btn btn-sm btn-danger" data-del="' + j.id + '">Remove</button>' : ''])) : empty('No labels saved yet.');
  });
  document.querySelectorAll('[data-del]').forEach(b => b.onclick = async () => { await api('/api/ir/judgments', {id: +b.dataset.del}, 'DELETE'); loadJ(); });
}
$('j-save').onclick = async () => {
  try {
    const ids = [...$('j-list').querySelectorAll('input:checked')].map(c => +c.value);
    await api('/api/ir/judgments', {query: need('j-q', 'Query'), project_ids: ids}); $('j-q').value = ''; loadJ();
  } catch (e) { $('j-out').innerHTML = '<div class="err">' + esc(e.message) + '</div>'; }
};

// ---- Crawler
$('c-go').onclick = () => run('c-out', 'c-go', async () => {
  const d = await api('/api/ir/crawl', {url: need('c-u', 'URL')});
  return '<h3>' + esc(d.title) + '</h3><p class="muted">' + esc(d.url) + '</p><p>' + esc(d.text.slice(0, 1200)) + (d.text.length > 1200 ? '…' : '') + '</p>' +
    '<h3>Links (' + d.links.length + ')</h3><div style="max-height:160px;overflow:auto" class="mono">' + d.links.map(esc).join('<br>') + '</div>' +
    '<h3>Keywords</h3>' + chips(d.extraction.keywords.map(k => k.keyword)) +
    (d.jsonld.length ? '<h3>JSON-LD</h3><pre class="mono">' + esc(JSON.stringify(d.jsonld, null, 1).slice(0, 2000)) + '</pre>' : '');
});

// ---- Extraction / semantic
const showIE = e => '<h3>Keywords</h3>' + chips(e.keywords.map(k => k.keyword + ' (' + k.count + ')')) + '<h3>Phrases</h3>' + chips(e.phrases.map(k => k.phrase + ' (' + k.count + ')')) +
  '<h3>Entities</h3>' + chips(e.entities.map(x => x.text + ' · ' + x.type)) + '<h3>JSON-LD</h3>' + (e.jsonld.length ? table(['Type', 'Name', 'Description', 'URL', 'Author'], e.jsonld.map(j => [j.type, j.name, j.description, j.url, j.author].map(esc))) : '<span class="muted">none found</span>');
$('x-go').onclick = () => run('x-out', 'x-go', async () => {
  if (!$('x-t').value.trim() && !$('x-h').value.trim()) throw Error('Enter text or HTML.');
  return showIE((await api('/api/ir/extract', {text: $('x-t').value, html: $('x-h').value})).extraction);
});
$('x-db').onclick = () => run('x-out', 'x-db', async () => showIE((await api('/api/ir/project-extraction')).extraction));
$('m-go').onclick = () => run('m-out', 'm-go', async () => {
  const d = await api('/api/ir/semantic', {query: need('m-q', 'Query')});
  if (d.error) return '<div class="info">' + esc(d.error) + '</div>';
  return d.results.length ? table(['Rank', 'Title', 'Latent cosine'], d.results.map(r => [r.position, esc(r.title), r.score])) : empty('No semantically related titles.');
});

// ---- MapReduce
$('mr-go').onclick = () => run('mr-out', 'mr-go', async () => {
  const docs = $('mr-d').value.split('\n').map(s => s.trim()).filter(Boolean);
  const d = await api('/api/ir/mapreduce', docs.length ? {documents: docs} : {});
  if (!d.terms.length) return empty(d.message || 'No terms.');
  return '<div class="info">' + d.documents + ' docs · ' + d.total_terms + ' terms · ' + d.unique_terms + ' unique. ' + esc(d.note) + '</div>' +
    table(['Term', 'Count', 'TF'], d.terms.map(t => [esc(t.term), t.count, t.tf]));
});

(async function init() {
  try { user = (await api('/api/auth/me')).user; } catch { return; }
  $('who').textContent = user.name + ' (' + user.role + ')';
  $('back').href = user.role === 'faculty' ? '/faculty' : '/student';
  if (user.role !== 'faculty') { $('c-go').disabled = true; }
  $('side').innerHTML = tabs.map(t => '<button data-t="' + t[0] + '">' + t[1] + '</button>').join('');
  document.querySelectorAll('#side button').forEach(b => b.onclick = () => show(b.dataset.t));
  try { projects = (await api('/api/ir/projects')).projects; } catch {}
  show(tabs.some(t => '#' + t[0] === location.hash) ? location.hash.slice(1) : 'search');
})();
