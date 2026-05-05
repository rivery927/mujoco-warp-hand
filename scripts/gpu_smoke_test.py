"""GPU smoke test: 验证 sm_120 上 Warp kernel 能正常编译并执行."""
import warp as wp
import numpy as np

wp.init()


@wp.kernel
def double_it(x: wp.array(dtype=float), y: wp.array(dtype=float)):
    i = wp.tid()
    y[i] = x[i] * 2.0


def main():
    n = 1024
    x_data = np.arange(n, dtype=np.float32)

    x = wp.array(x_data, device="cuda:0")
    y = wp.zeros(n, dtype=float, device="cuda:0")

    wp.launch(kernel=double_it, dim=n, inputs=[x, y], device="cuda:0")
    wp.synchronize()

    result = y.numpy()
    expected = x_data * 2.0
    match = np.allclose(result, expected)

    print(f"GPU kernel 执行: {'✅ 成功' if match else '❌ 失败'}")
    print(f"前 5 个输入:  {x_data[:5]}")
    print(f"前 5 个输出:  {result[:5]}")
    print(f"前 5 个期望:  {expected[:5]}")


if __name__ == "__main__":
    main()
