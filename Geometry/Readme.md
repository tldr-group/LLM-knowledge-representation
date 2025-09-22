# LLM Knowledge Representation - Geometry Intervention

A convenient and systematic approach for coordinate transformations in LLM knowledge representation experiments.

## 🚀 Quick Start

```python
# Simply change this line in CONFIG:
CONFIG['transformation_type'] = 'polar_3d'  # Choose any transformation type
```

### Step 1: Choose Your Transformation

Edit the `CONFIG` in `intervention.py`:

```python
CONFIG = {
    'transformation_type': 'polar_3d',  # Change this to any available type
    # ... all other settings remain the same
}
```

### Step 2: Run Your Experiment

```bash
python Geometry/intervention.py
```

## 🔧 Available Geometry Shapes

| Transformation Type | Description | Dimensions |
|-------------------|-------------|------------|
| `polar_3d` | 3D polar coordinates: [r×cos(θ), r×sin(θ), r] | 3D |
| `polar_2d` | 2D polar coordinates: [r×cos(θ), r×sin(θ)] | 2D |
| `polar_with_period` | Polar + period: [r×cos(θ), r×sin(θ), period] | 3D |
| `unit_circle_3d` | Unit circle + radius: [cos(θ), sin(θ), r] | 3D |
| `unit_circle_period` | Unit circle + period: [cos(θ), sin(θ), period] | 3D |
| `atomic_only` | Atomic number only: [atomic_number] | 1D |
| `cartesian_3d` | Cartesian coordinates: [atomic_number, group, period] | 3D |
| `random_control` | Random control: [random_value] | 1D |
| `random_polar` | Random polar: [cos(random_θ), sin(random_θ), r] | 3D |
| `scaled_polar` | Scaled polar: [cos(α), sin(θ), √period] | 3D |
| `mixed_coordinates` | Mixed coordinates: [group/cos(θ), sin(θ), r] | 3D |

**Where:**
- θ = group × (2π/18) (group-based angle)
- r = atomic_number (atomic number as radius)  

