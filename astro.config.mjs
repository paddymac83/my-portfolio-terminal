// @ts-check
import { defineConfig } from 'astro/config';
import sitemap from '@astrojs/sitemap';
import rehypeRaw from 'rehype-raw';

// https://astro.build/config
export default defineConfig({
  site: 'https://dennisklappe.github.io',
  base: '/',
  integrations: [sitemap()],
  markdown: {
    shikiConfig: {
      theme: 'css-variables',
      langs: [],
      wrap: true,
    },
    remarkRehype: {
      allowDangerousHtml: true,
    },
    rehypePlugins: [rehypeRaw],
  },
});