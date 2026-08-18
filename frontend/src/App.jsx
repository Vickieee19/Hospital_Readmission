import React from 'react'
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import { AuthProvider } from './context/AuthContext'
import ProtectedRoute from './components/ProtectedRoute'
import StaffLoginPage from './pages/StaffLoginPage'
import AdminLoginPage from './pages/AdminLoginPage'
import AdminDashboardPage from './pages/AdminDashboardPage'
import ClinicalDashboardPage from './pages/ClinicalDashboardPage'

export default function App() {
  return (
    <AuthProvider>
      <BrowserRouter>
        <Routes>
          {/* Public Authentication Pages */}
          <Route path="/login" element={<StaffLoginPage />} />
          
          {/* Admin Login: Unlinked route accessible solely by direct URL */}
          <Route path="/admin-login" element={<AdminLoginPage />} />

          {/* Protected Administrator Dashboard */}
          <Route
            path="/admin"
            element={
              <ProtectedRoute requiredRole="admin">
                <AdminDashboardPage />
              </ProtectedRoute>
            }
          />

          {/* Protected Clinical Decision Support Dashboard (Nurse & Admin) */}
          <Route
            path="/"
            element={
              <ProtectedRoute>
                <ClinicalDashboardPage />
              </ProtectedRoute>
            }
          />

          {/* Fallback */}
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </BrowserRouter>
    </AuthProvider>
  )
}
