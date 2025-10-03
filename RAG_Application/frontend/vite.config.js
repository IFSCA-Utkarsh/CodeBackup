import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import path from 'path';

export default defineConfig({
  server: {
    port: 3000,
  },
  preview: {
    port: 3000,
  },
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
      crypto: path.resolve(__dirname, 'node_modules/crypto-browserify'),  // Ensure this is here
    },
  },
  optimizeDeps: {
    include: ['crypto-browserify'], // Include the crypto-browserify for optimization
  },
  plugins: [react()],
});
