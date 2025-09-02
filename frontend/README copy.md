# Curate - Coming Soon

A simple Next.js app with TypeScript, React, and Tailwind CSS displaying a "Coming Soon" page.

## Features

- ⚡ Next.js 14 with App Router
- 🎨 Tailwind CSS for styling
- 📱 Responsive design
- 🚀 Optimized for Vercel deployment
- 💻 TypeScript support

## Development

```bash
# Install dependencies
npm install

# Start development server
npm run dev

# Build for production
npm run build

# Start production server
npm start
```

## Deployment on Vercel

### Option 1: GitHub Integration (Recommended)

1. Push your code to a GitHub repository
2. Visit [vercel.com](https://vercel.com)
3. Sign in with GitHub
4. Click "New Project"
5. Import your repository
6. Vercel will automatically detect Next.js and deploy

### Option 2: Vercel CLI

1. Install Vercel CLI globally:
   ```bash
   npm i -g vercel
   ```

2. Login to Vercel:
   ```bash
   vercel login
   ```

3. Deploy:
   ```bash
   # For preview deployment
   npm run preview
   
   # For production deployment
   npm run deploy
   ```

### Option 3: Drag and Drop

1. Run `npm run build`
2. Visit [vercel.com](https://vercel.com)
3. Drag and drop the `.next` folder to the deployment area

## Project Structure

```
├── src/
│   └── app/
│       ├── globals.css      # Tailwind CSS imports
│       ├── layout.tsx       # Root layout
│       └── page.tsx         # Main "Coming Soon" page
├── next.config.js           # Next.js configuration
├── tailwind.config.ts       # Tailwind CSS configuration
├── vercel.json             # Vercel deployment configuration
└── package.json            # Dependencies and scripts
```

## Configuration

The project is pre-configured for optimal Vercel deployment with:

- Standalone output for better performance
- Compression enabled
- Image optimization
- Static page generation where possible

## License

MIT
