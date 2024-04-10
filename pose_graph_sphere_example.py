import g2o
import numpy as np
from jaxlie import SO3, SE3


def main():
    solver = g2o.OptimizationAlgorithmLevenberg(g2o.BlockSolverSE3(g2o.LinearSolverEigenSE3()))
    optimizer = g2o.SparseOptimizer()
    optimizer.set_algorithm(solver)
    optimizer.set_verbose(True)

    with open("sphere.g2o", "r") as f:
        vertex_cnt, edge_cnt = 0, 0
        for line in f.readlines():
            arr = line.split()

            if arr[0] == "VERTEX_SE3:QUAT":
                idx = int(arr[1])
                se3 = SE3.from_rotation_and_translation(
                    rotation=SO3.from_quaternion_xyzw(np.array(arr[5:], dtype=np.float32)),
                    translation=np.array(arr[2:5], dtype=np.float32)
                )

                v = g2o.VertexSE3()
                v.set_id(idx)
                v.set_estimate(g2o.Isometry3d(se3.as_matrix()))

                optimizer.add_vertex(v)
                if idx == 0:
                    v.set_fixed(True)
                vertex_cnt += 1

            elif arr[0] == "EDGE_SE3:QUAT":
                idx1, idx2 = int(arr[1]), int(arr[2])
                se3 = SE3.from_rotation_and_translation(
                    rotation=SO3.from_quaternion_xyzw(np.array(arr[6:10], dtype=np.float32)),
                    translation=np.array(arr[3:6], dtype=np.float32)
                )
                info = np.zeros((6, 6))
                i_u, j_u = np.triu_indices(6)
                i_l, j_l = np.tril_indices(6, k=-1)
                info[i_u, j_u] = arr[10:]
                info[i_l, j_l] = info[j_l, i_l]

                e = g2o.EdgeSE3()
                e.set_id(edge_cnt)
                e.set_vertex(0, optimizer.vertex(idx1))
                e.set_vertex(1, optimizer.vertex(idx2))
                e.set_measurement(g2o.Isometry3d(se3.as_matrix()))
                e.set_information(info)

                optimizer.add_edge(e)
                edge_cnt += 1

    print(f"Total: {vertex_cnt} vertices | {edge_cnt} edges")
    optimizer.initialize_optimization()
    optimizer.optimize(15)

    optimizer.save("result.g2o")
    from plotly import graph_objects as go
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(
        x=[p.estimate().position()[0] for _, p in optimizer.vertices().items()],
        y=[p.estimate().position()[1] for _, p in optimizer.vertices().items()],
        z=[p.estimate().position()[2] for _, p in optimizer.vertices().items()],
        mode="markers"
    ))
    fig.show()


if __name__ == "__main__":
    main()
