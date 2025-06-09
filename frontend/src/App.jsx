import ClaimForm from './components/ClaimForm'

function App() {
  return (
    <div className="min-h-screen bg-gray-100 p-6 font-sans">
      <header className="mb-6">
        <h1 className="text-2xl font-bold text-center text-blue-800">
          Insurance Claim Fraud Detection
        </h1>
        <p className="text-center text-sm text-gray-600">
          Submit new claims to assess fraud risk in real-time.
        </p>
      </header>

      <main className="max-w-5xl mx-auto">
        <ClaimForm />
      </main>

    </div>
  )
}

export default App
