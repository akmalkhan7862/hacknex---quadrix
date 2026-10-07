/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    './app/**/*.{js,ts,jsx,tsx,mdx}',
    './pages/**/*.{js,ts,jsx,tsx,mdx}',
    './components/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {
      colors: {
        paper: 'var(--paper)',
        'paper-2': 'var(--paper-2)',
        surface: 'var(--surface)',
        ink: 'var(--ink)',
        'ink-soft': 'var(--ink-soft)',
        rule: 'var(--rule)',
        accent: 'var(--accent)',
        mark: 'var(--mark)',
        ok: 'var(--ok)',
        warn: 'var(--warn)',
        bad: 'var(--bad)',
      },
      fontFamily: {
        serif: ['Newsreader', 'Georgia', 'serif'],
        sans: ['IBM Plex Sans', '-apple-system', 'sans-serif'],
        mono: ['IBM Plex Mono', 'monospace'],
      },
      borderRadius: {
        DEFAULT: '4px',
        sm: '2px',
        md: '4px',
        lg: '4px',
      }
    },
  },
  plugins: [],
};
