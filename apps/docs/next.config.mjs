import { createMDX } from 'fumadocs-mdx/next';

const withMDX = createMDX();

export default withMDX({
  reactStrictMode: true,
  output: 'export',
  trailingSlash: true,
  images: { unoptimized: true },
  turbopack: {},
  webpack(config) {
    // Fumadocs loads generated MDX through a dynamic file URL that webpack cannot track for
    // persistent cache invalidation. This docs app is small, so disabling that cache keeps builds
    // deterministic and warning-free without affecting Turbopack development.
    config.cache = false;
    return config;
  },
});
