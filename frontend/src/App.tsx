import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import Layout from '@/components/Layout'
import Dashboard from '@/pages/Dashboard'
import PatientAnalysis from '@/pages/PatientAnalysis'
import BatchAnalysis from '@/pages/BatchAnalysis'
import Results from '@/pages/Results'
import ModelManagement from '@/pages/ModelManagement'
import Settings from '@/pages/Settings'

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Layout />}>
          <Route index element={<Navigate to="/dashboard" replace />} />
          <Route path="dashboard" element={<Dashboard />} />
          <Route path="patients" element={<PatientAnalysis />} />
          <Route path="patients/:code" element={<PatientAnalysis />} />
          <Route path="batch" element={<BatchAnalysis />} />
          <Route path="results" element={<Results />} />
          <Route path="models" element={<ModelManagement />} />
          <Route path="settings" element={<Settings />} />
        </Route>
      </Routes>
    </BrowserRouter>
  )
}
