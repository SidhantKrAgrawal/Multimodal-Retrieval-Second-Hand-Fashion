/** @type {import('tailwindcss').Config} */
module.exports = {
    content: [
      "./src/app/**/*.{js,ts,jsx,tsx}",         // App directory (required)
      "./pages/**/*.{js,ts,jsx,tsx}",       // Optional: if you use pages
      "./components/**/*.{js,ts,jsx,tsx}",  // If you put UI in /components
      "./src/app/recommend/**/*.{js,ts,jsx,tsx}"
    ],
    theme: {
      extend: {},
    },
    plugins: [],
  };
  