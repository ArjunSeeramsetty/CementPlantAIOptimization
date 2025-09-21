# 🔐 GitHub Secrets Setup Guide

## Setting up GCP Service Account Key as GitHub Secret

Follow these steps to add your GCP service account key to GitHub Secrets:

### Step 1: Go to GitHub Repository Settings
1. Navigate to your repository: `https://github.com/ArjunSeeramsetty/CementPlantAIOptimization`
2. Click on **Settings** tab (at the top of the repository page)
3. In the left sidebar, click on **Secrets and variables** → **Actions**

### Step 2: Add Repository Secret
1. Click **New repository secret**
2. Set the following values:
   - **Name**: `GCP_SA_KEY`
   - **Secret**: Copy the entire contents of `.secrets/cement-ops-key.json`

### Step 3: Copy the Service Account Key Content
Run this command to copy the key content:

```bash
# On Windows PowerShell
Get-Content .secrets/cement-ops-key.json | Set-Clipboard

# On Linux/Mac
cat .secrets/cement-ops-key.json | pbcopy
```

### Step 4: Paste in GitHub
1. Paste the copied JSON content into the **Secret** field in GitHub
2. Click **Add secret**

### Step 5: Verify the Secret
The secret should now appear in your repository secrets list as:
- **Name**: GCP_SA_KEY
- **Last updated**: [Current timestamp]

## Alternative: Using GitHub CLI

If you have GitHub CLI installed, you can set the secret directly:

```bash
# Install GitHub CLI first if not installed
# Then run:
gh secret set GCP_SA_KEY < .secrets/cement-ops-key.json
```

## Verification

After setting up the secret, you can:

1. **Trigger the deployment workflow** by pushing to the main branch
2. **Manually trigger** by going to Actions → GCP Deployment → Run workflow
3. **Check the workflow logs** to ensure the secret is being used correctly

## Security Notes

- ✅ The service account key is encrypted in GitHub Secrets
- ✅ It's only accessible to GitHub Actions workflows
- ✅ It's not visible in logs or pull requests
- ✅ You can rotate the key anytime by updating the secret

## Next Steps

Once the GitHub secret is set up:

1. **Push your changes** to trigger the deployment
2. **Monitor the deployment** in GitHub Actions
3. **Check the deployed application** at the provided URL
4. **Set up monitoring** and alerting as needed

---

**🎯 Your deployment will be available at a URL like:**
`https://cement-digital-twin-[hash]-uc.a.run.app`

**📊 Monitor deployment progress at:**
`https://github.com/ArjunSeeramsetty/CementPlantAIOptimization/actions`
