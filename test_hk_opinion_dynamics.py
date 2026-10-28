import unittest

import networkx as nx
import numpy as np

import hk_opinion_dynamics as hk


class TestHegselmannKrauseModel(unittest.TestCase):
    def test_four_agent_example(self):
        graph = nx.Graph([(0, 1), (2, 3)])
        initial = np.array([0.10, 0.25, 0.70, 0.80])
        result = hk.hk_step(graph, initial, epsilon=0.20)
        np.testing.assert_allclose(result, [0.175, 0.175, 0.75, 0.75])

    def test_complete_graph_and_large_epsilon(self):
        graph = nx.complete_graph(3)
        initial = np.array([0.2, 0.5, 0.8])
        result = hk.hk_step(graph, initial, epsilon=1.0)
        np.testing.assert_allclose(result, [0.5, 0.5, 0.5])

    def test_zero_epsilon_does_not_change_distinct_opinions(self):
        graph = nx.complete_graph(4)
        initial = np.array([0.1, 0.10001, 0.6, 0.9])
        result = hk.hk_step(graph, initial, epsilon=0.0)
        np.testing.assert_allclose(result, initial)

    def test_cluster_count(self):
        values = np.array([0.1, 0.1002, 0.7, 0.9, 0.9001])
        self.assertEqual(hk.count_clusters(values, cluster_tol=1e-3), 3)

    def test_simulation_converges_on_complete_graph(self):
        graph = nx.complete_graph(5)
        initial = np.array([0.05, 0.2, 0.45, 0.7, 0.95])
        result = hk.simulate(graph, initial, epsilon=1.0, tol=1e-10, max_iter=10)
        self.assertTrue(result.converged)
        self.assertLess(np.ptp(result.final_opinions), 1e-10)


if __name__ == "__main__":
    unittest.main()
