import math
import os
import tempfile
import unittest

import app as appmod
from ir import automaton, extraction, graph, ranking, recommend, parallel
from ir.crawler import validate_url

TITLES = [
    {"id": 1, "title": "AI Based Attendance System using Face Recognition", "status": "submitted"},
    {"id": 2, "title": "Blockchain Based Land Registry and Verification", "status": "approved"},
    {"id": 3, "title": "Face Detection for Smart Attendance Management", "status": "submitted"},
    {"id": 4, "title": "Crop Disease Detection with Deep Learning", "status": "submitted"},
]


class Algorithms(unittest.TestCase):
    def test_methods_rank_relevant_first(self):
        for m in ranking.METHODS:
            r = ranking.rank("face attendance", TITLES, m)
            self.assertIn(r[0]["id"], (1, 3), m)
            self.assertEqual(r[0]["position"], 1)

    def test_lm_laplace_is_valid_log_probability(self):
        s = ranking.lm_scores("blockchain zzzunseen", [t["title"] for t in TITLES])
        self.assertTrue(all(math.isfinite(x) and x < 0 for x in s))
        self.assertEqual(max(range(4), key=lambda i: s[i]), 1)

    def test_dfr_nonnegative_and_zero_for_nomatch(self):
        s = ranking.dfr_scores("blockchain", [t["title"] for t in TITLES])
        self.assertTrue(all(x >= 0 for x in s))
        self.assertGreater(s[1], 0)
        self.assertEqual(s[0], 0)

    def test_empty_inputs(self):
        self.assertEqual(ranking.rank("x", [], "tfidf"), [])
        self.assertEqual(ranking.rank("the", TITLES, "dfr"), [])

    def test_metrics(self):
        self.assertEqual(ranking.precision_at_k([1, 2, 3], {1, 3}, 2), 0.5)
        self.assertEqual(ranking.recall_at_k([1, 2, 3], {1, 3}, 2), 0.5)
        self.assertAlmostEqual(ranking.average_precision([1, 2, 3], {1, 3}), (1 + 2 / 3) / 2)
        ev = ranking.evaluate({"face attendance": {1, 3}}, TITLES, "tfidf", 2)
        self.assertEqual(ev["MAP"], 1.0)
        self.assertIsNone(ranking.evaluate({}, TITLES, "lm"))

    def test_passages(self):
        r = ranking.rank_passages("blockchain ledger", "Cooking pasta at home.\n\nA blockchain is a ledger.")
        self.assertEqual(r[0]["id"], 2)

    def test_automaton(self):
        d = automaton.match(["he", "she", "hers"], "ushers")
        self.assertEqual({m["term"] for m in d["matches"]}, {"he", "she", "hers"})

    def test_pagerank(self):
        r = graph.rank_projects(TITLES, 0.1)
        self.assertAlmostEqual(sum(x["pagerank"] for x in r["results"]), 1.0, places=3)
        self.assertEqual(graph.rank_projects([], 0.1)["results"], [])
        iso = graph.rank_projects([{"id": 1, "title": "alpha"}, {"id": 2, "title": "beta"}], 0.5)
        self.assertEqual(iso["edges"], 0)
        self.assertAlmostEqual(iso["results"][0]["pagerank"], 0.5, places=4)

    def test_recommend(self):
        r = recommend.content_based(TITLES[0]["title"], TITLES, 3, exclude_id=1)
        self.assertTrue(all(x["id"] != 1 for x in r))
        self.assertEqual(r[0]["id"], 3)
        recs, msg = recommend.collaborative([(1, 1, 5)], 1)
        self.assertEqual(recs, []); self.assertIn("Not enough", msg)
        k = recommend.knowledge_based(TITLES, status="approved", include=["blockchain"])
        self.assertEqual([x["id"] for x in k], [2])

    def test_collaborative_works_with_data(self):
        ratings = [(1, 1, 5), (1, 2, 4), (2, 1, 5), (2, 2, 4), (2, 3, 5), (3, 1, 4), (3, 3, 5), (3, 4, 1)]
        recs, _ = recommend.collaborative(ratings, 1)
        self.assertEqual(recs[0]["project_id"], 3)

    def test_extraction(self):
        html = '<script type="application/ld+json">{"@type":"Article","name":"Hi","author":{"name":"Ann"}}</script><p>Visit Google Research in 2024 about NLP.</p>'
        e = extraction.extract_all("", html)
        self.assertEqual(e["jsonld"][0]["author"], "Ann")
        types = {x["type"] for x in e["entities"]}
        self.assertTrue({"year", "acronym", "proper_noun"} <= types)
        self.assertEqual(extraction.semantic_search("x", TITLES[:2])["results"], [])

    def test_mapreduce(self):
        d = parallel.mapreduce_term_frequency(["face face id", "face"])
        self.assertEqual(d["terms"][0]["term"], "face"); self.assertEqual(d["terms"][0]["count"], 3)

    def test_url_validation_blocks_internal(self):
        for u in ("http://localhost/", "http://127.0.0.1/", "http://10.0.0.5/", "http://169.254.169.254/",
                  "http://[::1]/", "ftp://example.com/", "http://example.com:8080/", "http://u:p@example.com/"):
            with self.assertRaises(ValueError, msg=u):
                validate_url(u)


class Routes(unittest.TestCase):
    def setUp(self):
        fd, self.path = tempfile.mkstemp(suffix=".db"); os.close(fd)
        appmod.DB_PATH = self.path
        appmod.init_db()
        appmod.app.config["TESTING"] = True
        self.c = appmod.app.test_client()

    def tearDown(self):
        os.remove(self.path)

    def reg(self, client, roll, name):
        r = client.post("/api/auth/register", json={"username": roll, "password": "pass123", "name": name})
        self.assertEqual(r.status_code, 201)

    def test_flow(self):
        self.assertEqual(self.c.post("/api/ir/search", json={"query": "x"}).status_code, 401)
        self.reg(self.c, "21CS001", "Alice")
        r = self.c.post("/api/student/project", json={"title": TITLES[0]["title"]})
        self.assertEqual(r.status_code, 201)
        s = self.c.post("/api/ir/search", json={"query": "face", "method": "dfr"}).get_json()
        self.assertEqual(s["results"][0]["position"], 1)
        self.assertEqual(self.c.post("/api/ir/search", json={"query": ""}).status_code, 400)
        self.assertEqual(self.c.post("/api/ir/search", json={"query": "face", "method": "bad"}).status_code, 400)
        self.assertEqual(self.c.post("/api/ir/search", json={"query": "face", "method": "all"}).status_code, 200)
        self.assertEqual(self.c.post("/api/ir/crawl", json={"url": "http://127.0.0.1"}).status_code, 403)
        # second student: submission response carries recommendations (own project excluded)
        c2 = appmod.app.test_client(); self.reg(c2, "21CS002", "Bob")
        r = c2.post("/api/student/project", json={"title": "Blockchain Based Land Registry and Verification"})
        self.assertEqual(r.status_code, 201)
        self.assertIn("recommendations", r.get_json())
        mine = c2.get("/api/ir/my-recommendations").get_json()
        self.assertTrue(all(x["title"] != "Blockchain Based Land Registry and Verification" for x in mine["recommendations"]))
        self.assertIn("Not enough", c2.get("/api/ir/recommend/collaborative").get_json()["message"])
        self.assertEqual(c2.post("/api/ir/ratings", json={"project_id": 2, "rating": 5}).status_code, 400)
        self.assertEqual(c2.post("/api/ir/ratings", json={"project_id": 1, "rating": 5}).status_code, 200)
        # evaluation: no labels -> honest message; faculty labels -> metrics
        self.assertEqual(c2.post("/api/ir/evaluate", json={}).get_json()["evaluations"], [])
        self.assertEqual(c2.post("/api/ir/judgments", json={"query": "face", "project_ids": [1]}).status_code, 403)
        f = appmod.app.test_client()
        self.assertEqual(f.post("/api/auth/login", json={"username": "faculty", "password": "faculty123"}).status_code, 200)
        self.assertEqual(f.post("/api/ir/judgments", json={"query": "face", "project_ids": [1]}).status_code, 201)
        ev = f.post("/api/ir/evaluate", json={"k": 1}).get_json()
        self.assertEqual(len(ev["evaluations"]), 3)
        self.assertEqual(f.post("/api/ir/crawl", json={"url": "http://localhost:3000"}).status_code, 400)
        self.assertEqual(f.post("/api/ir/pagerank", json={"threshold": 2}).status_code, 400)
        self.assertEqual(f.post("/api/ir/automaton", json={"terms": "face", "text": "a face"}).status_code, 200)


if __name__ == "__main__":
    unittest.main()
