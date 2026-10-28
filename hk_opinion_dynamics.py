"""
Моделирование сетевого варианта модели Хегсельманна–Краузе.

По умолчанию скрипт:
1) моделирует траектории мнений на полном графе;
2) сравнивает число кластеров и число итераций до сходимости
   для полного графа, случайного графа Эрдёша–Реньи и кольцевой решётки;
3) сохраняет графики, CSV с результатами и конфигурацию эксперимента.

Запуск:
    python hk_opinion_dynamics.py --output results
"""

from __future__ import annotations
import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


GRAPH_NAMES = {
    "complete": "Полный граф",
    "erdos_renyi": "Случайный граф Эрдёша–Реньи",
    "ring_lattice": "Регулярная кольцевая решётка",
}


@dataclass
class SimulationResult:
    history: np.ndarray
    converged: bool
    iterations: int
    max_change: float
    final_opinions: np.ndarray
    graph_components: int


def generate_graph(
    graph_type: str,
    n: int,
    seed: int = 42,
    ring_degree: int = 4,
    erdos_renyi_p: float | None = None,
) -> nx.Graph:
    """Создаёт неориентированный граф одного из трёх поддерживаемых типов.

    Для сравнения с кольцевой решёткой по умолчанию выбирается вероятность
    ребра p = ring_degree / (n - 1), то есть ожидаемая степень близка к
    степени узла в решётке.
    """
    if n < 2:
        raise ValueError("Количество агентов n должно быть не меньше 2.")

    if graph_type == "complete":
        graph = nx.complete_graph(n)
    elif graph_type == "erdos_renyi":
        if erdos_renyi_p is None:
            erdos_renyi_p = min(1.0, ring_degree / (n - 1))
        if not 0.0 <= erdos_renyi_p <= 1.0:
            raise ValueError("Вероятность ребра должна принадлежать [0, 1].")
        graph = nx.erdos_renyi_graph(n, erdos_renyi_p, seed=seed)
        graph.graph["edge_probability"] = erdos_renyi_p
    elif graph_type == "ring_lattice":
        if ring_degree < 2 or ring_degree >= n or ring_degree % 2 != 0:
            raise ValueError(
                "Для кольцевой решётки ring_degree должен быть чётным числом "
                "в диапазоне от 2 до n - 1."
            )
        graph = nx.watts_strogatz_graph(n, k=ring_degree, p=0, seed=seed)
    else:
        raise ValueError(
            f"Неизвестный тип графа: {graph_type}. "
            f"Допустимые типы: {', '.join(GRAPH_NAMES)}."
        )

    graph.graph["graph_type"] = graph_type
    graph.graph["n"] = n
    return graph


def _adjacency_matrix(graph: nx.Graph) -> np.ndarray:
    """Создаёт булеву матрицу потенциальных связей и включает диагональ."""
    if graph.is_directed():
        raise ValueError(
            "Эта реализация рассчитана на неориентированные графы. "
            "Для ориентированной сети нужно отдельно задать направление влияния."
        )
    if graph.number_of_nodes() == 0:
        raise ValueError("Граф не должен быть пустым.")

    nodes = list(graph.nodes())
    adjacency = nx.to_numpy_array(graph, nodelist=nodes, dtype=bool)
    np.fill_diagonal(adjacency, True)
    return adjacency


def _hk_step_with_adjacency(
    adjacency: np.ndarray,
    opinions: np.ndarray,
    epsilon: float,
) -> np.ndarray:
    """Векторизованное обновление мнений при заранее построенной матрице связей."""
    confidence = np.abs(opinions[:, None] - opinions[None, :]) <= epsilon
    trusted = adjacency & confidence
    trusted_counts = trusted.sum(axis=1)
    return (trusted @ opinions) / trusted_counts


def hk_step(
    graph: nx.Graph,
    opinions: Iterable[float] | np.ndarray,
    epsilon: float,
) -> np.ndarray:
    """Выполняет один синхронный шаг сетевого варианта модели Хегсельманна–Краузе.

    Агент учитывает собственное мнение и мнения соседей по исходному графу,
    если абсолютная разность мнений не превышает epsilon. Все новые мнения
    рассчитываются на основе старого вектора одновременно.
    """
    if epsilon < 0:
        raise ValueError("Радиус доверия epsilon должен быть неотрицательным.")

    x = np.asarray(opinions, dtype=float)
    if x.ndim != 1:
        raise ValueError("Вектор мнений должен быть одномерным.")
    if len(x) != graph.number_of_nodes():
        raise ValueError("Длина вектора мнений должна совпадать с числом узлов графа.")
    if not np.all(np.isfinite(x)):
        raise ValueError("Вектор мнений содержит нечисловые или бесконечные значения.")

    adjacency = _adjacency_matrix(graph)
    return _hk_step_with_adjacency(adjacency, x, epsilon)

def simulate(
    graph: nx.Graph,
    initial_opinions: Iterable[float] | np.ndarray,
    epsilon: float,
    tol: float = 1e-6,
    max_iter: int = 5000,
) -> SimulationResult:
    """Запускает симуляцию до сходимости или достижения max_iter."""
    if epsilon < 0:
        raise ValueError("Радиус доверия epsilon должен быть неотрицательным.")
    if tol <= 0:
        raise ValueError("Параметр tol должен быть положительным.")
    if max_iter < 1:
        raise ValueError("max_iter должен быть не меньше 1.")

    x = np.asarray(initial_opinions, dtype=float).copy()
    if x.ndim != 1:
        raise ValueError("Вектор начальных мнений должен быть одномерным.")
    if len(x) != graph.number_of_nodes():
        raise ValueError("Длина начального вектора должна совпадать с числом узлов графа.")
    if not np.all(np.isfinite(x)):
        raise ValueError("Вектор начальных мнений содержит нечисловые или бесконечные значения.")
    if np.any(x < 0) or np.any(x > 1):
        raise ValueError("В рамках этой работы начальные мнения должны принадлежать [0, 1].")

    adjacency = _adjacency_matrix(graph)
    history = [x.copy()]
    converged = False
    max_change = float("inf")
    iterations = 0

    for iteration in range(1, max_iter + 1):
        next_x = _hk_step_with_adjacency(adjacency, x, epsilon)
        max_change = float(np.max(np.abs(next_x - x)))
        history.append(next_x.copy())
        iterations = iteration
        x = next_x

        if max_change < tol:
            converged = True
            break

    return SimulationResult(
        history=np.vstack(history),
        converged=converged,
        iterations=iterations,
        max_change=max_change,
        final_opinions=x.copy(),
        graph_components=nx.number_connected_components(graph),
    )


def count_clusters(opinions: Iterable[float] | np.ndarray, cluster_tol: float = 1e-3) -> int:
    """Оценивает число кластеров итоговых мнений.

    Мнения сортируются; новый кластер начинается, когда разность между двумя
    соседними отсортированными значениями превышает cluster_tol. Это численный
    критерий группировки мнений, а не радиус доверия модели.
    """
    if cluster_tol < 0:
        raise ValueError("cluster_tol должен быть неотрицательным.")

    values = np.asarray(opinions, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("Для подсчёта кластеров требуется непустой одномерный вектор.")
    if not np.all(np.isfinite(values)):
        raise ValueError("Вектор мнений содержит нечисловые или бесконечные значения.")

    ordered = np.sort(values)
    return int(1 + np.count_nonzero(np.diff(ordered) > cluster_tol))


def classify_outcome(cluster_count: int) -> str:
    if cluster_count == 1:
        return "consensus"
    if cluster_count == 2:
        return "polarization"
    return "fragmentation"


def run_epsilon_experiment(
    graph: nx.Graph,
    graph_type: str,
    initial_opinions: np.ndarray,
    epsilon_values: Iterable[float],
    tol: float,
    max_iter: int,
    cluster_tol: float,
) -> list[dict[str, object]]:
    """Запускает серию симуляций на одном графе для разных радиусов доверия."""
    rows: list[dict[str, object]] = []
    for epsilon in epsilon_values:
        result = simulate(graph, initial_opinions, float(epsilon), tol=tol, max_iter=max_iter)
        # При epsilon=0 агенты взаимодействуют только с участниками с точно
        # совпадающими мнениями. Не объединяем просто близкие, но разные мнения.
        effective_cluster_tol = 0.0 if float(epsilon) == 0.0 else cluster_tol
        clusters = count_clusters(result.final_opinions, cluster_tol=effective_cluster_tol)
        rows.append(
            {
                "graph_type": graph_type,
                "graph_name": GRAPH_NAMES[graph_type],
                "epsilon": float(epsilon),
                "cluster_count": clusters,
                "outcome": classify_outcome(clusters),
                "iterations": result.iterations,
                "converged": result.converged,
                "max_change": result.max_change,
                "final_min": float(np.min(result.final_opinions)),
                "final_max": float(np.max(result.final_opinions)),
                "final_range": float(np.ptp(result.final_opinions)),
                "graph_components": result.graph_components,
                "n_agents": graph.number_of_nodes(),
                "n_edges": graph.number_of_edges(),
            }
        )
    return rows


def save_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_trajectory_csv(path: Path, history: np.ndarray) -> None:
    n_agents = history.shape[1]
    columns = ["iteration"] + [f"agent_{i + 1}" for i in range(n_agents)]
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(columns)
        for t, opinions in enumerate(history):
            writer.writerow([t, *[float(value) for value in opinions]])


def plot_trajectories(
    path: Path,
    history: np.ndarray,
    graph_name: str,
    epsilon: float,
    converged: bool,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    times = np.arange(history.shape[0])
    for agent_index in range(history.shape[1]):
        ax.plot(times, history[:, agent_index], linewidth=0.9, alpha=0.65)
    ax.set_title(
        f"Траектории мнений: {graph_name}, ε = {epsilon:.3f}\n"
        f"Сходимость достигнута: {'да' if converged else 'нет'}"
    )
    ax.set_xlabel("Итерация t")
    ax.set_ylabel("Мнение агента xᵢ(t)")
    ax.set_ylim(-0.03, 1.03)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_clusters_by_epsilon(path: Path, rows: list[dict[str, object]]) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for graph_type in GRAPH_NAMES:
        selected = [row for row in rows if row["graph_type"] == graph_type]
        selected.sort(key=lambda row: float(row["epsilon"]))
        ax.plot(
            [float(row["epsilon"]) for row in selected],
            [int(row["cluster_count"]) for row in selected],
            marker="o",
            markersize=3,
            linewidth=1.5,
            label=GRAPH_NAMES[graph_type],
        )
    ax.set_title("Количество итоговых кластеров в зависимости от радиуса доверия")
    ax.set_xlabel("Радиус доверия ε")
    ax.set_ylabel("Количество кластеров")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_iterations_by_epsilon(path: Path, rows: list[dict[str, object]], max_iter: int) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for graph_type in GRAPH_NAMES:
        selected = [row for row in rows if row["graph_type"] == graph_type]
        selected.sort(key=lambda row: float(row["epsilon"]))
        ax.plot(
            [float(row["epsilon"]) for row in selected],
            [int(row["iterations"]) for row in selected],
            marker="o",
            markersize=3,
            linewidth=1.5,
            label=GRAPH_NAMES[graph_type],
        )
    ax.axhline(max_iter, linestyle="--", linewidth=1, label="Лимит итераций")
    ax.set_title("Число итераций до выполнения критерия сходимости")
    ax.set_xlabel("Радиус доверия ε")
    ax.set_ylabel("Число итераций")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Симуляция сетевой модели Хегсельманна–Краузе и вычислительные эксперименты."
    )
    parser.add_argument("--n", type=int, default=100, help="Количество агентов (по умолчанию: 100).")
    parser.add_argument("--seed", type=int, default=42, help="Начальное значение генератора случайных чисел.")
    parser.add_argument("--tol", type=float, default=1e-6, help="Допуск для критерия сходимости.")
    parser.add_argument("--cluster-tol", type=float, default=1e-3, help="Допуск для группировки итоговых мнений в кластеры.")
    parser.add_argument("--max-iter", type=int, default=5000, help="Максимальное число итераций.")
    parser.add_argument("--eps-points", type=int, default=51, help="Число значений ε на отрезке [0, 1].")
    parser.add_argument("--epsilon-demo", type=float, default=0.25, help="Радиус доверия для графика траекторий.")
    parser.add_argument("--ring-degree", type=int, default=4, help="Степень узла кольцевой решётки; должно быть чётным числом.")
    parser.add_argument("--er-p", type=float, default=None, help="Вероятность ребра графа Эрдёша–Реньи. По умолчанию подбирается по ring-degree.")
    parser.add_argument("--output", type=Path, default=Path("hk_results"), help="Папка для сохранения файлов результатов.")
    args = parser.parse_args()

    if args.n < 2:
        parser.error("--n должен быть не меньше 2.")
    if args.eps_points < 2:
        parser.error("--eps-points должен быть не меньше 2.")
    if not 0 <= args.epsilon_demo <= 1:
        parser.error("--epsilon-demo должен принадлежать [0, 1].")
    if args.max_iter < 1 or args.tol <= 0 or args.cluster_tol < 0:
        parser.error("Проверьте значения --max-iter, --tol и --cluster-tol.")

    output_dir: Path = args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    initial_opinions = rng.random(args.n)
    epsilon_values = np.linspace(0.0, 1.0, args.eps_points)

    all_rows: list[dict[str, object]] = []
    graphs: dict[str, nx.Graph] = {}

    for graph_type in GRAPH_NAMES:
        graph = generate_graph(
            graph_type,
            args.n,
            seed=args.seed,
            ring_degree=args.ring_degree,
            erdos_renyi_p=args.er_p,
        )
        graphs[graph_type] = graph
        all_rows.extend(
            run_epsilon_experiment(
                graph=graph,
                graph_type=graph_type,
                initial_opinions=initial_opinions,
                epsilon_values=epsilon_values,
                tol=args.tol,
                max_iter=args.max_iter,
                cluster_tol=args.cluster_tol,
            )
        )

    save_csv(output_dir / "experiment_summary.csv", all_rows)
    plot_clusters_by_epsilon(output_dir / "clusters_by_epsilon.png", all_rows)
    plot_iterations_by_epsilon(output_dir / "iterations_by_epsilon.png", all_rows, args.max_iter)

    demo_graph = graphs["complete"]
    demo_result = simulate(
        demo_graph,
        initial_opinions,
        epsilon=args.epsilon_demo,
        tol=args.tol,
        max_iter=args.max_iter,
    )
    plot_trajectories(
        output_dir / "opinion_trajectories.png",
        demo_result.history,
        GRAPH_NAMES["complete"],
        args.epsilon_demo,
        demo_result.converged,
    )
    save_trajectory_csv(output_dir / "opinion_trajectories.csv", demo_result.history)

    graph_info = {}
    for graph_type, graph in graphs.items():
        graph_info[graph_type] = {
            "name": GRAPH_NAMES[graph_type],
            "nodes": graph.number_of_nodes(),
            "edges": graph.number_of_edges(),
            "connected_components": nx.number_connected_components(graph),
            "average_degree": float(np.mean([degree for _, degree in graph.degree()])),
        }
        if graph_type == "erdos_renyi":
            graph_info[graph_type]["edge_probability"] = float(
                graph.graph.get("edge_probability", args.er_p if args.er_p is not None else args.ring_degree / (args.n - 1))
            )

    configuration = {
        "model": "network-constrained Hegselmann-Krause bounded-confidence model",
        "n_agents": args.n,
        "seed": args.seed,
        "convergence_tolerance": args.tol,
        "cluster_tolerance": args.cluster_tol,
        "max_iterations": args.max_iter,
        "epsilon_values": [float(value) for value in epsilon_values],
        "trajectory_demo_epsilon": args.epsilon_demo,
        "ring_degree": args.ring_degree,
        "graphs": graph_info,
        "trajectory_demo": {
            "converged": demo_result.converged,
            "iterations": demo_result.iterations,
            "cluster_count": count_clusters(demo_result.final_opinions, args.cluster_tol),
            "max_change_at_stop": demo_result.max_change,
        },
        "note": (
            "Для каждого агента в множество доверия входят он сам и соседи по исходному графу, "
            "чьи мнения отличаются не более чем на epsilon. Обновления выполняются синхронно. "
            "Параметр cluster_tolerance используется только для численного подсчёта кластеров."
        ),
    }
    with (output_dir / "experiment_config.json").open("w", encoding="utf-8") as file:
        json.dump(configuration, file, ensure_ascii=False, indent=2)

    not_converged = sum(not bool(row["converged"]) for row in all_rows)
    print("Эксперименты завершены.")
    print(f"Результаты сохранены в: {output_dir.resolve()}")
    print(f"Агентов: {args.n}; графов: {len(graphs)}; значений epsilon: {len(epsilon_values)}")
    print(f"Всего запусков в сравнении: {len(all_rows)}")
    print(f"Запусков, не достигших критерия сходимости за лимит: {not_converged}")
    print("Созданы файлы:")
    for filename in (
        "opinion_trajectories.png",
        "clusters_by_epsilon.png",
        "iterations_by_epsilon.png",
        "experiment_summary.csv",
        "opinion_trajectories.csv",
        "experiment_config.json",
    ):
        print(f"  - {filename}")


if __name__ == "__main__":
    main()
