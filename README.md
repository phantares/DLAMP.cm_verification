# Verification for DLAMP.cm

This repo uses **two separate environments**, because `wrf-python` cannot be
installed into the main project environment — it requires legacy
`numpy.distutils` and is incompatible with `numpy>=2.0` and Python 3.13,
which the main project depends on.

| Environment | Manager | Purpose | Activate |
|---|---|---|---|
| `.venv` | `uv` | Main project | `source .venv/bin/activate` |
| `wrf-env` | `mamba` | dBZ calculation only (`wrf-python`) | `mamba activate wrf-env` |

---

## 1. Main Environment (`uv`)

This project uses `uv` for fast dependency management.

**Install `uv`** (skip if already installed):
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Sync the project environment** to install all required packages:
```bash
uv sync
```

---

## 2. Secondary Environment: wrf-python (`mamba`)

This environment is used only for radar reflectivity (dBZ) calculations via
`wrf-python`, since it can't coexist with the main environment's dependencies.

**Install mamba** (skip if already installed):
```bash
curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj bin/micromamba
```

**Create the environment:**
```bash
mamba create -n wrf-env python=3.10 -y
mamba activate wrf-env
mamba install -c conda-forge wrf-python netcdf4 xarray h5py pyyaml python-dotenv -y
```

**Running scripts that depend on wrf-python:**

Scripts under `scripts/` that use `wrf-python` (e.g. `scripts/write_dbz.py`) must
be run with the `wrf-env` environment:

```bash
mamba activate wrf-env
python scripts/write_dbz.py exp_name
```

---

## 3. Configuration

Set up your environment variables. Copy the example file to a new `.env` file:
```bash
cp .env.example .env
```

Edit paths in `.env` using your preferred editor:
```bash
vi .env
```