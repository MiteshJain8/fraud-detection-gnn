import { useState } from 'react'
import reactLogo from './assets/react.svg'
import viteLogo from '/vite.svg'
// import './App.css' // Remove App.css import

function App() {
  const [count, setCount] = useState(0)

  return (
    // Apply Tailwind classes for max width, horizontal centering, padding, and text alignment
    <div className="max-w-5xl mx-auto p-8 text-center">
      <div className="flex justify-center space-x-4 mb-8"> {/* Flex container for logos */}
        <a href="https://vite.dev" target="_blank" rel="noreferrer">
          {/* Apply Tailwind classes for logo size, padding, transitions and hover effects */}
          <img src={viteLogo} className="h-24 p-6 transition-filter duration-300 hover:drop-shadow-[0_0_2em_#646cffaa]" alt="Vite logo" />
        </a>
        <a href="https://react.dev" target="_blank" rel="noreferrer">
          {/* Apply Tailwind classes for logo size, padding, transitions, hover effects and animation */}
          <img src={reactLogo} className="h-24 p-6 transition-filter duration-300 hover:drop-shadow-[0_0_2em_#61dafbaa] motion-safe:animate-spin-slow" alt="React logo" />
        </a>
      </div>
      <h1 className="text-5xl font-bold mb-4">Vite + React</h1> {/* Tailwind for heading */}
      <div className="p-8"> {/* Tailwind for card padding */}
        <button
          className="rounded-lg border border-transparent px-5 py-2.5 text-base font-medium bg-[#1a1a1a] cursor-pointer transition-colors duration-200 hover:border-[#646cff] focus:outline-none focus:ring-4 focus:ring-[#646cff33] dark:bg-[#f9f9f9] dark:text-[#213547] dark:hover:border-[#747bff] dark:focus:ring-[#747bff33] mb-4" // Tailwind for button styling
          onClick={() => setCount((count) => count + 1)}
        >
          count is {count}
        </button>
        <p className="mb-4"> {/* Tailwind for paragraph margin */}
          Edit <code className="font-mono bg-gray-700/50 p-1 rounded">src/App.jsx</code> and save to test HMR
        </p>
      </div>
      <p className="text-[#888888]"> {/* Tailwind for text color */}
        Click on the Vite and React logos to learn more
      </p>
    </div>
  )
}

// (Removed commented-out custom animation code)

export default App
