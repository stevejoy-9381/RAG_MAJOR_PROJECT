/** @type {import('next').NextConfig} */
const nextConfig = {
  output: 'standalone',
  transpilePackages: ['recharts', 'victory-vendor'],
}

export default nextConfig
