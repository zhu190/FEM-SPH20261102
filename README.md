# SPH_CESHI

Julia / CUDA SPH 源代码及 soil 示例参数配置。

## 目录

- `src41/`：SPH 求解器源代码。
- `eg/main.jl`：示例运行入口。
- `eg/soil.xml`：示例参数配置。

本仓库仅包含代码、XML 配置和说明，不包含 `.dat` 初始粒子数据、计算结果、缓存及 `.bak` 备份。源文件和配置保持原样。

## 依赖与运行

需要 Julia 和支持 CUDA 的 NVIDIA GPU。源目录未提供 `Project.toml` / `Manifest.toml`；请在所用 Julia 环境中准备 `CUDA`、`StaticArrays`、`LightXML` 和 `Adapt`。

运行 soil 示例前，请在本地将所需的 `soil_geom.dat`（初始粒子几何）和 `soil_state.dat`（初始粒子状态）放入 `eg/`。这两个文件未上传，已由 `.gitignore` 排除。

从仓库根目录运行：

```powershell
julia eg/main.jl eg/soil 0
```

`0` 为 GPU 设备编号。示例结果写入 `eg/soil/`，同样不纳入版本控制。

本次上传未执行完整仿真。
