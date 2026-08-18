import React from 'react'
import { Navigate, useLocation } from 'react-router-dom'
import { useAuth } from '../context/AuthContext'
import { Stethoscope } from 'lucide-react'

export default function ProtectedRoute({ children, requiredRole = null }) {
  const { user, isAuthenticated, loading } = useAuth()
  const location = useLocation()

  if (loading) {
    return (
      <div className="min-h-screen flex flex-col items-center justify-center bg-slate-50 text-slate-700">
        <div className="flex h-14 w-14 items-center justify-center rounded-2xl bg-blue-50 text-blue-600 border border-blue-100 shadow-sm animate-pulse mb-4">
          <Stethoscope className="h-7 w-7" />
        </div>
        <div className="h-6 w-6 animate-spin rounded-full border-2 border-blue-600 border-t-transparent mb-2"></div>
        <p className="text-xs font-semibold text-slate-500">Verifying secure clinical credentials...</p>
      </div>
    )
  }

  if (!isAuthenticated) {
    // Redirect unauthenticated requests to /login
    return <Navigate to="/login" state={{ from: location }} replace />
  }

  if (requiredRole) {
    if (requiredRole === 'admin' && user?.role !== 'admin' && user?.role !== 'sub_admin') {
      return <Navigate to="/" replace />
    } else if (requiredRole !== 'admin' && user?.role !== requiredRole) {
      return <Navigate to="/" replace />
    }
  }

  return children
}
