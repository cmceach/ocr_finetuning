# Deploying Label Studio on Azure

This guide covers multiple deployment options for Label Studio on Azure, suitable for different use cases and requirements.

## Deployment Options

### 1. Azure Container Instances (ACI) - Quick Start
**Best for**: Development, testing, simple deployments
**Pros**: Fast deployment, pay-per-use, no infrastructure management
**Cons**: Limited scalability, data persistence requires Azure Storage

### 2. Azure App Service - Managed Platform
**Best for**: Production workloads, automatic scaling, managed infrastructure
**Pros**: Auto-scaling, SSL certificates, integrated monitoring
**Cons**: Higher cost, less control over infrastructure

### 3. Azure Kubernetes Service (AKS) - Enterprise Scale
**Best for**: Large-scale deployments, high availability, advanced networking
**Pros**: Full Kubernetes features, high scalability, production-ready
**Cons**: Complex setup, requires Kubernetes knowledge

## Prerequisites

- Azure account with active subscription
- Azure CLI installed and configured
- Docker (for custom images, optional)

## Option 1: Azure Container Instances (ACI)

### Quick Deployment

```bash
# Make script executable
chmod +x deployment/azure/label_studio_aci_deploy.sh

# Deploy Label Studio
./deployment/azure/label_studio_aci_deploy.sh \
    --resource-group rg-label-studio \
    --container-name label-studio \
    --cpu 2 \
    --memory 4
```

### With Persistent Storage

```bash
# First, set up Azure Storage
./deployment/azure/label_studio_storage_setup.sh \
    --resource-group rg-label-studio \
    --storage-account labelstudiodata \
    --container-name label-studio-data

# Then deploy with storage configuration
az container create \
    --resource-group rg-label-studio \
    --name label-studio \
    --image heartexlabs/label-studio:latest \
    --cpu 2 \
    --memory 4Gi \
    --ip-address Public \
    --ports 8080 \
    --environment-variables \
        LABEL_STUDIO_HOST="http://<container-ip>:8080" \
        STORAGE_TYPE=azure \
        STORAGE_AZURE_ACCOUNT_NAME=labelstudiodata \
        STORAGE_AZURE_ACCOUNT_KEY="<storage-key>" \
        STORAGE_AZURE_CONTAINER_NAME=label-studio-data
```

### Access Your Deployment

After deployment, get the container IP:
```bash
az container show \
    --resource-group rg-label-studio \
    --name label-studio \
    --query ipAddress.ip --output tsv
```

Access Label Studio at: `http://<container-ip>:8080`

### Management Commands

```bash
# View logs
az container logs --resource-group rg-label-studio --name label-studio --follow

# Stop container
az container stop --resource-group rg-label-studio --name label-studio

# Restart container
az container restart --resource-group rg-label-studio --name label-studio

# Delete container
az container delete --resource-group rg-label-studio --name label-studio --yes
```

## Option 2: Azure App Service

### Deployment Steps

```bash
# Make script executable
chmod +x deployment/azure/label_studio_app_service_deploy.sh

# Deploy to App Service
./deployment/azure/label_studio_app_service_deploy.sh \
    --resource-group rg-label-studio \
    --app-name label-studio-prod \
    --sku B1
```

### App Service SKU Options

- **B1 (Basic)**: $13/month - Development/testing
- **S1 (Standard)**: $70/month - Production workloads
- **P1V2 (Premium)**: $146/month - High-performance

### Custom Domain and SSL

```bash
# Add custom domain
az webapp config hostname add \
    --webapp-name label-studio-prod \
    --resource-group rg-label-studio \
    --hostname labelstudio.yourdomain.com

# Enable SSL (requires App Service Certificate or Let's Encrypt)
az webapp config ssl bind \
    --name label-studio-prod \
    --resource-group rg-label-studio \
    --certificate-thumbprint <thumbprint> \
    --ssl-type SNI
```

### Scaling

```bash
# Scale up (change instance size)
az appservice plan update \
    --name label-studio-plan \
    --resource-group rg-label-studio \
    --sku S1

# Scale out (add instances)
az webapp scale \
    --resource-group rg-label-studio \
    --name label-studio-prod \
    --instance-count 3
```

## Option 3: Azure Kubernetes Service (AKS)

### Prerequisites

- AKS cluster created
- kubectl configured
- Helm installed

### Deploy with Helm

```bash
# Add Label Studio Helm repository (if available)
helm repo add label-studio https://charts.labelstud.io
helm repo update

# Deploy Label Studio
helm install label-studio label-studio/label-studio \
    --namespace label-studio \
    --create-namespace \
    --set ingress.enabled=true \
    --set ingress.hosts[0].host=labelstudio.yourdomain.com \
    --set persistence.enabled=true \
    --set persistence.size=50Gi
```

### Manual Kubernetes Deployment

Create `label-studio-k8s.yaml`:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: label-studio
  namespace: label-studio
spec:
  replicas: 2
  selector:
    matchLabels:
      app: label-studio
  template:
    metadata:
      labels:
        app: label-studio
    spec:
      containers:
      - name: label-studio
        image: heartexlabs/label-studio:latest
        ports:
        - containerPort: 8080
        env:
        - name: LABEL_STUDIO_HOST
          value: "https://labelstudio.yourdomain.com"
        volumeMounts:
        - name: label-studio-data
          mountPath: /label-studio/data
      volumes:
      - name: label-studio-data
        azureFile:
          secretName: azure-file-secret
          shareName: label-studio-share
          readOnly: false
---
apiVersion: v1
kind: Service
metadata:
  name: label-studio-service
  namespace: label-studio
spec:
  selector:
    app: label-studio
  ports:
  - port: 80
    targetPort: 8080
  type: LoadBalancer
```

Deploy:
```bash
kubectl apply -f label-studio-k8s.yaml
```

## Azure Storage Integration

### Set Up Azure Storage

```bash
./deployment/azure/label_studio_storage_setup.sh \
    --resource-group rg-label-studio \
    --storage-account labelstudiodata \
    --container-name label-studio-data
```

### Configure Label Studio to Use Azure Storage

Set these environment variables:

```bash
STORAGE_TYPE=azure
STORAGE_AZURE_ACCOUNT_NAME=labelstudiodata
STORAGE_AZURE_ACCOUNT_KEY=<your-storage-key>
STORAGE_AZURE_CONTAINER_NAME=label-studio-data
```

### Using Azure Files (for AKS)

```bash
# Create Azure File Share
az storage share create \
    --name label-studio-share \
    --account-name labelstudiodata \
    --account-key <storage-key>

# Create Kubernetes secret
kubectl create secret generic azure-file-secret \
    --from-literal=azurestorageaccountname=labelstudiodata \
    --from-literal=azurestorageaccountkey=<storage-key>
```

## Environment Variables

### Required Variables

- `LABEL_STUDIO_HOST`: Full URL where Label Studio is accessible
- `LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED`: Enable local file serving (set to `true`)

### Optional Variables

- `LABEL_STUDIO_USERNAME`: Default admin username
- `LABEL_STUDIO_PASSWORD`: Default admin password
- `LABEL_STUDIO_DATABASE`: Database connection string (for PostgreSQL)
- `STORAGE_TYPE`: Storage backend (`local`, `azure`, `s3`, `gcs`)
- `REDIS_URL`: Redis connection string (for task queue)

## Security Best Practices

1. **Use Azure Key Vault** for sensitive credentials:
   ```bash
   az keyvault create --name label-studio-vault --resource-group rg-label-studio
   az keyvault secret set --vault-name label-studio-vault --name storage-key --value <key>
   ```

2. **Enable HTTPS** for all production deployments

3. **Configure Network Security Groups** to restrict access

4. **Use Managed Identity** instead of storage keys when possible

5. **Enable Azure Monitor** for logging and monitoring

## Monitoring and Logging

### Azure Monitor Integration

```bash
# Enable Application Insights
az monitor app-insights component create \
    --app label-studio-insights \
    --location eastus \
    --resource-group rg-label-studio

# Get instrumentation key
INSTRUMENTATION_KEY=$(az monitor app-insights component show \
    --app label-studio-insights \
    --resource-group rg-label-studio \
    --query instrumentationKey -o tsv)
```

### View Logs

**ACI:**
```bash
az container logs --resource-group rg-label-studio --name label-studio --follow
```

**App Service:**
```bash
az webapp log tail --resource-group rg-label-studio --name label-studio-prod
```

**AKS:**
```bash
kubectl logs -f deployment/label-studio -n label-studio
```

## Cost Optimization

### ACI
- **Estimated cost**: ~$0.000012/second (~$30/month for 2 CPU, 4GB RAM, running 24/7)
- **Tips**: Stop containers when not in use, use smaller sizes for dev

### App Service
- **Estimated cost**: $13-146/month depending on SKU
- **Tips**: Use Basic tier for dev, Standard for production, enable auto-shutdown

### AKS
- **Estimated cost**: $73/month (cluster) + VM costs
- **Tips**: Use spot instances for non-critical workloads, scale down during off-hours

## Troubleshooting

### Container Won't Start
- Check logs: `az container logs --resource-group <rg> --name <name>`
- Verify resource limits (CPU/memory)
- Check environment variables

### Can't Access Label Studio
- Verify port is exposed (8080)
- Check network security groups
- Verify LABEL_STUDIO_HOST matches actual URL

### Data Persistence Issues
- Ensure Azure Storage is properly configured
- Check storage account permissions
- Verify container/file share exists

### Performance Issues
- Scale up resources (CPU/memory)
- Use Azure Files instead of Blob Storage for better performance
- Enable caching if using Azure Storage

## Integration with OCR Pipeline

After deploying Label Studio on Azure, update your `.env` file:

```bash
LABEL_STUDIO_URL=https://label-studio-prod.azurewebsites.net
LABEL_STUDIO_API_KEY=<your-api-key>
```

Then use it with your OCR pipeline:

```python
from src.preprocessing.azure_di_preprocessor import AzureDIPreprocessor

preprocessor = AzureDIPreprocessor()
preannotations = preprocessor.preannotate_batch(
    document_paths=["doc1.pdf"],
    image_urls=["https://label-studio-prod.azurewebsites.net/data/upload/doc1.pdf"]
)
```

## Next Steps

1. Choose your deployment option based on requirements
2. Set up Azure Storage for data persistence
3. Configure custom domain and SSL (production)
4. Set up monitoring and alerts
5. Integrate with your OCR fine-tuning pipeline

For more information, see:
- [Label Studio Documentation](https://labelstud.io/guide/)
- [Azure Container Instances](https://docs.microsoft.com/azure/container-instances/)
- [Azure App Service](https://docs.microsoft.com/azure/app-service/)

