# CodeFlare SDK Patterns for Ray in OpenShift AI

Based on the official demo-notebooks, this document outlines the correct patterns for working with Ray clusters in OpenShift AI.

## Key Findings

✅ **Use CodeFlare SDK** - Not raw kubectl/YAML for cluster management  
✅ **Use kube-authkit** - For authentication  
✅ **Programmatic cluster creation** - Via Python SDK, not YAML files  
✅ **Job submission** - Use Ray Job Submission Client or interactive ray.init()  

## Standard Pattern

### 1. Import and Authentication

```python
from codeflare_sdk import Cluster, ClusterConfiguration, set_api_client
from kube_authkit import AuthConfig, get_k8s_client

# Get your token with: oc whoami -t
auth_config = AuthConfig(
    method="openshift",
    k8s_api_host="https://api.example.com:6443",
    token="sha256~XXXXX",  # oc whoami -t
)

api_client = get_k8s_client(config=auth_config)
set_api_client(api_client)
```

### 2. Configure and Create Cluster

```python
cluster = Cluster(ClusterConfiguration(
    name='my-cluster',
    head_cpu_requests=1,
    head_cpu_limits=1,
    head_memory_requests=6,
    head_memory_limits=8,
    head_extended_resource_requests={'nvidia.com/gpu': 1},  # For GPU
    worker_extended_resource_requests={'nvidia.com/gpu': 1},
    num_workers=2,
    worker_cpu_requests='250m',
    worker_cpu_limits=1,
    worker_memory_requests=4,
    worker_memory_limits=6,
    # image="custom-ray-image",  # Optional
    write_to_file=False,
))

# Create cluster
cluster.apply()
cluster.wait_ready()
```

### 3a. Interactive Mode (for development)

```python
import ray

# Get cluster connection info
ray_cluster_uri = cluster.cluster_uri()
ray_dashboard_uri = cluster.cluster_dashboard_uri()

# Connect to cluster
runtime_env = {"pip": ["transformers", "datasets"]}
ray.init(address=ray_cluster_uri, runtime_env=runtime_env)

# Define and run remote functions
@ray.remote
def train_fn():
    # Your training code
    pass

# Execute
ray.get(train_fn.remote())
```

### 3b. Job Submission Mode (for production)

```python
# Get job client
client = cluster.job_client

# Submit job
submission_id = client.submit_job(
    entrypoint="python train.py",
    runtime_env={
        "working_dir": "./",
        "pip": "requirements.txt"
    },
)

# Monitor job
client.get_job_logs(submission_id)
client.get_job_status(submission_id)
```

### 4. Cleanup

```python
cluster.down()
# No explicit logout needed - kube-authkit manages automatically
```

## Default Ray Images

The CodeFlare SDK automatically selects images based on Python version:

- **Python 3.11**: `quay.io/modh/ray:2.52.1-py311-cu121`
- **Python 3.12**: `quay.io/modh/ray:2.54.1-py312-cu128`

Override with `image="custom-image"` in ClusterConfiguration if needed.

## GPU Support

For GPU workloads:

```python
head_extended_resource_requests={'nvidia.com/gpu': 1}
worker_extended_resource_requests={'nvidia.com/gpu': 1}
```

## Storage for Multi-Node Training

When running distributed training, use persistent storage accessible across nodes:

```python
from ray.train import RunConfig

ray_trainer = TorchTrainer(
    train_func,
    scaling_config=ScalingConfig(...),
    run_config=RunConfig(storage_path="s3://bucket/path"),
)
```

See: [S3-compatible storage docs](https://www.github.com/project-codeflare/codeflare-sdk/tree/main/docs/s3-compatible-storage.md)

## What NOT to Do

❌ Don't use raw kubectl commands for cluster management  
❌ Don't manually create RayCluster YAML files  
❌ Don't use static cluster definitions  
❌ Don't manage authentication manually  

The CodeFlare SDK abstracts all of this away.

## Differences from Our Current Approach

### Current (churn demo)
```bash
# Manual YAML files
kubectl apply -f ray_cluster.yaml
kubectl apply -f ray_service.yaml

# Manual connection
python train.py --ray-address ray://cluster:10001
```

### Correct (CodeFlare SDK)
```python
# Programmatic cluster creation
cluster = Cluster(ClusterConfiguration(...))
cluster.apply()
cluster.wait_ready()

# SDK-managed connection
ray_uri = cluster.cluster_uri()
ray.init(address=ray_uri)
```

## Migration Strategy for Churn Demo

To align with OpenShift AI best practices, we should:

1. **Replace** `tests/test_ray_cluster.ipynb` to use CodeFlare SDK
2. **Update** main demo notebook to use SDK instead of kubectl
3. **Keep** standalone scripts (`0-generate_data.py`, etc.) for flexibility
4. **Add** notebook version using CodeFlare SDK for OpenShift AI users
5. **Keep** `ray_cluster.yaml` as reference for manual deployments

## References

- **demo-notebooks/guided-demos/2_basic_interactive.ipynb** - Interactive mode
- **demo-notebooks/guided-demos/1_cluster_job_client.ipynb** - Job submission
- **CodeFlare SDK Docs**: https://github.com/project-codeflare/codeflare-sdk
- **kube-authkit**: Authentication management
