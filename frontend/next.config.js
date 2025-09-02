/** @type {import('next').NextConfig} */
const nextConfig = {
  output: 'standalone',
  trailingSlash: false,
  images: {
    unoptimized: false,
  },
  poweredByHeader: false,
  compress: true,
}

module.exports = nextConfig
