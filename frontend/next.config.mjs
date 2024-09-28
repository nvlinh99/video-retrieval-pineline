import path from "path";

/** @type {import('next').NextConfig} */
const nextConfig = {
  typescript: {
    ignoreBuildErrors: true,
  },
  eslint: {
    ignoreDuringBuilds: true,
  },
  webpack(config) {
    config.resolve.alias.unfetch = path.resolve(
      path.resolve(),
      "node_modules/unfetch/dist/unfetch.mjs"
    );
    return config;
  },
  images: {
    domains: ["picsum.photos"],
  },
};

export default nextConfig;
