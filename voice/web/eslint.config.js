import js from '@eslint/js'
import globals from 'globals'
import reactHooks from 'eslint-plugin-react-hooks'
import reactRefresh from 'eslint-plugin-react-refresh'
import { defineConfig, globalIgnores } from 'eslint/config'

export default defineConfig([
  globalIgnores(['dist']),
  {
    files: ['**/*.{js,jsx}'],
    ignores: ['public/**'],
    extends: [
      js.configs.recommended,
      reactHooks.configs.flat.recommended,
      reactRefresh.configs.vite,
    ],
    languageOptions: {
      globals: globals.browser,
      parserOptions: { ecmaFeatures: { jsx: true } },
    },
  },
  {
    // AudioWorklet scripts run in AudioWorkletGlobalScope, not the browser
    // or worker globals eslint knows about.
    files: ['public/**/*.js'],
    extends: [js.configs.recommended],
    languageOptions: {
      globals: { ...globals.worker, AudioWorkletProcessor: 'readonly', registerProcessor: 'readonly' },
    },
  },
])
