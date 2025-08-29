# AVA-TAR Deployment Guide

## Deploying to Render

### Prerequisites
1. A GitHub account
2. A Render account (sign up at https://render.com)
3. Your Flask app code pushed to a GitHub repository

### Step-by-Step Deployment

#### 1. Prepare Your Repository
- Ensure all files are committed to your GitHub repository
- Make sure your `requirements.txt` file is up to date
- Verify your model files are included in the repository

#### 2. Connect to Render
1. Log in to your Render dashboard
2. Click "New +" and select "Web Service"
3. Connect your GitHub account if not already connected
4. Select your repository

#### 3. Configure the Web Service
- **Name**: `ava-tar` (or your preferred name)
- **Environment**: `Python 3`
- **Build Command**: `pip install -r requirements.txt`
- **Start Command**: `gunicorn app:app`
- **Plan**: Choose based on your needs:
  - **Starter**: $7/month (good for testing)
  - **Standard**: $25/month (recommended for production)

#### 4. Environment Variables (Optional)
Add these if needed:
- `FLASK_ENV`: `production`
- `SECRET_KEY`: Generate a secure secret key

#### 5. Deploy
1. Click "Create Web Service"
2. Render will automatically build and deploy your app
3. Monitor the build logs for any issues
4. Once deployed, you'll get a URL like `https://ava-tar.onrender.com`

### Important Notes

#### Model Files
- Ensure your model files (`model/Rat&Mouse_LV_RN34_best.pth`) are included in the repository
- These files are large, so consider using Git LFS if your repository becomes too large

#### File Storage
- Render provides ephemeral storage (files are lost on restart)
- For production, consider using external storage services like AWS S3 for uploaded files
- The current setup uses local storage which works for development but may not be suitable for production

#### Performance Considerations
- The free tier has limitations on build time and memory
- PyTorch models can be memory-intensive
- Consider using smaller models or optimizing for production

### Troubleshooting

#### Common Issues
1. **Build Failures**: Check the build logs for missing dependencies
2. **Memory Issues**: Upgrade to a higher tier plan
3. **Model Loading Errors**: Ensure model files are properly included in the repository

#### Logs
- View logs in the Render dashboard under your service
- Check both build logs and runtime logs for errors

### Alternative Deployment Options

#### Using render.yaml (Infrastructure as Code)
1. Ensure `render.yaml` is in your repository root
2. In Render dashboard, select "New +" → "Blueprint"
3. Connect your repository
4. Render will automatically configure the service based on the YAML file

#### Manual Configuration
If you prefer not to use the YAML file:
1. Follow the web service setup above
2. Manually configure all settings in the Render dashboard

### Post-Deployment

#### Testing
1. Test file upload functionality
2. Verify 3D reconstruction works
3. Check that all routes are accessible
4. Test the volume analysis features

#### Monitoring
- Set up health checks
- Monitor resource usage
- Set up alerts for downtime

### Security Considerations
- Change the secret key in production
- Consider adding authentication if needed
- Validate all file uploads
- Implement rate limiting for production use
