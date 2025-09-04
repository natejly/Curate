# Curate Frontend

A Next.js application for dataset management and ML training visualization.

## Deployment

This app is configured for deployment on Vercel with Next.js 13+ App Router.

### Build Configuration
- **Framework**: Next.js
- **Build Command**: `npm run build`
- **Output Directory**: `.next` (automatic)
- **Node Version**: 18.x or later

### Environment Variables
Set the following environment variables in Vercel:
- `BACKEND_URL`: Your backend API URL (e.g., `https://your-backend.vercel.app`)

### Routes
- `/` - Main dashboard with dataset upload
- `/training-console` - Live training console with metrics
- `/api/upload` - File upload endpoint

## Development

```bash
npm install
npm run dev
```

## Build

```bash
npm run build
npm start
```