# Ray Cluster Test

This directory contains a test to verify your Ray cluster setup before running the full demo.

## Test File

**`test_cluster.ipynb`** - Creates a test Ray cluster, shows dashboard URL, waits for you to verify, then deletes it.

## Quick Start

1. **Open the notebook:**
   ```
   tests/test_cluster.ipynb
   ```

2. **Update the configuration** (top of Step 1):
   - `NAMESPACE` - Your OpenShift namespace (default: `user-mhepburn`)
   - `CLUSTER_NAME` - Name for test cluster (default: `test-ray-cluster`)
   - `NUM_WORKERS` - Number of GPU workers (default: `2`)
   - `USE_GPU` - Enable GPU support (default: `True`)

3. **Run all cells:**
   - Creates RayCluster resource
   - Waits for pods ready (2-5 minutes)
   - Shows dashboard URL and connection info
   - **Pauses for you to verify**
   - Press Enter to delete cluster
   - Verifies cleanup

## What This Tests

✅ KubeRay operator is installed and working  
✅ GPU nodes are available (if `USE_GPU=True`)  
✅ Ray images can be pulled  
✅ Cluster creation completes without errors  
✅ Workers join the cluster successfully  
✅ Cluster deletion cleans up all resources  

## Expected Timeline

- **Cluster creation:** 2-5 minutes (depends on image pull)
- **Manual verification:** As long as you want
- **Cluster deletion:** 10-30 seconds

## Success Criteria

You should see:

```
✓ RayCluster configuration prepared
✓ RayCluster created successfully!
✓ Ray cluster is ready! (took XXs)
✓ All pods cleaned up
```

## Troubleshooting

### "KubeRay operator not found"

Install the KubeRay operator:

```bash
kubectl apply -k "github.com/ray-project/kuberay/ray-operator/config/default"
```

### "No nodes with GPU available"

Check GPU node availability:

```bash
kubectl get nodes -l nvidia.com/gpu.present=true
```

Or disable GPU for testing by setting `USE_GPU = False` in the notebook.

### "Pods stuck in Pending"

Check pod events:

```bash
kubectl describe pod <pod-name> -n <your-namespace>
```

Common issues:
- **ImagePullBackOff:** Network issues or image not accessible
- **Insufficient GPU:** No GPU nodes available
- **Insufficient CPU/Memory:** Nodes at capacity

### "Timeout waiting for cluster"

The Ray image can be large. First pull may take 5-10 minutes.

Check pod status manually:

```bash
kubectl get pods -n <your-namespace> -l ray.io/cluster=test-ray-cluster -w
```

## After Testing

Once this test passes, you're ready to run the full demo:

1. **Upload training data:**
   ```bash
   python 0-generate_data.py --n 2000 --key churn/train.parquet
   ```

2. **Run training:**
   ```bash
   python 1-train.py
   ```

3. **Evaluate model:**
   ```bash
   python 2-evaluate.py
   ```

4. **Register model:**
   ```bash
   python 3-register.py
   ```

See the main README.md for full details.
