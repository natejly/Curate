# Vercel Deployment Guide for Curate

## Prerequisites
1. Vercel account (free tier available)
2. GitHub repository with your code
3. OpenAI API key

## Deployment Steps

### 1. Prepare Your Repository
Make sure all files are committed to your GitHub repository, including:
- `vercel.json` (configuration file)
- `requirements.txt` (Python dependencies)
- `api/index.py` (serverless function handler)

### 2. Connect to Vercel
1. Go to [vercel.com](https://vercel.com)
2. Sign up/login with GitHub
3. Click "New Project"
4. Import your Curate repository

### 3. Configure Environment Variables
In Vercel dashboard, add these environment variables:
- `OPENAI_API_KEY`: Your OpenAI API key

### 4. Deploy
1. Vercel will automatically deploy when you push to main branch
2. First deployment may take 5-10 minutes due to TensorFlow installation

## Important Notes

### Limitations on Vercel
- **File Storage**: Vercel is stateless - uploaded files and models won't persist between requests
- **Memory**: Limited to 1GB RAM (may not be enough for large models)
- **Execution Time**: Max 300 seconds per request
- **TensorFlow**: Large dependency, may cause cold starts

### Recommended Alternative: Railway or Render
For a full ML application with file persistence, consider:
- **Railway.app**: Better for Python apps with persistent storage
- **Render.com**: Free tier with persistent storage
- **Google Cloud Run**: Scalable container deployment

## Production Optimizations

### For Better Performance:
1. **Model Storage**: Use external storage (AWS S3, Google Cloud Storage)
2. **Caching**: Implement model caching strategies
3. **Smaller Models**: Use quantized or distilled models
4. **Background Tasks**: Move training to background workers

### Environment Variables Needed:
```
OPENAI_API_KEY=your_openai_key_here
NODE_ENV=production
```

## Alternative Deployment (Railway - Recommended)

Railway is better suited for ML apps:

1. **Install Railway CLI**:
```bash
npm install -g @railway/cli
```

2. **Login and Deploy**:
```bash
railway login
railway init
railway up
```

3. **Add Environment Variables** in Railway dashboard

Railway provides:
- Persistent file storage
- Better Python/ML support
- Automatic HTTPS
- Easy environment management

## Troubleshooting

### Common Issues:
1. **TensorFlow Installation Timeout**: Use lighter ML libraries or external model serving
2. **Memory Limits**: Reduce batch sizes or use model quantization
3. **Cold Starts**: Implement model warm-up strategies

### Logs:
- Check Vercel Function logs in dashboard
- Monitor performance and errors
