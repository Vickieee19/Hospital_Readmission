import React, { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { useAuth } from '../context/AuthContext'
import {
  ShieldAlert,
  Shield,
  Lock,
  KeyRound,
  AlertCircle,
  Sparkles,
  ArrowRight,
  Eye,
  EyeOff,
  Terminal,
} from 'lucide-react'

export default function AdminLoginPage() {
  const navigate = useNavigate()
  const { login, isAuthenticated, isAdmin } = useAuth()

  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [showPassword, setShowPassword] = useState(false)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  // If already logged in as admin, redirect directly to /admin
  React.useEffect(() => {
    if (isAuthenticated) {
      if (isAdmin) {
        navigate('/admin', { replace: true })
      } else {
        setError('You are currently signed in as a staff member. Please sign in with an Administrator account.')
      }
    }
  }, [isAuthenticated, isAdmin, navigate])

  const handleSubmit = async (e) => {
    e.preventDefault()
    setError('')

    const trimmedUser = username.trim()
    if (!trimmedUser || !password) {
      setError('Please enter administrator credentials.')
      return
    }

    setLoading(true)
    try {
      // Require admin role
      await login(trimmedUser, password, 'admin')
      navigate('/admin', { replace: true })
    } catch (err) {
      setError(err.message || 'Invalid administrator credentials. Access denied.')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-indigo-950 flex flex-col justify-center py-12 sm:px-6 lg:px-8 selection:bg-indigo-500 selection:text-white text-slate-100">
      <div className="sm:mx-auto sm:w-full sm:max-w-md text-center space-y-3">
        <div className="inline-flex h-16 w-16 items-center justify-center rounded-2xl bg-indigo-600 text-white shadow-xl shadow-indigo-500/25 border border-indigo-400/30 mb-1">
          <KeyRound className="h-9 w-9" />
        </div>
        <h1 className="text-2xl sm:text-3xl font-extrabold tracking-tight text-white">
          CareGrid Admin Console
        </h1>
        <p className="text-xs sm:text-sm text-slate-400 max-w-sm mx-auto">
          Restricted Security & Staff User Management Portal
        </p>
      </div>

      <div className="mt-8 sm:mx-auto sm:w-full sm:max-w-md px-4 sm:px-0">
        <div className="rounded-3xl border border-slate-700/80 bg-slate-800/90 p-8 sm:p-9 shadow-2xl shadow-black/50 backdrop-blur-md space-y-6">
          <div className="flex items-center justify-between border-b border-slate-700 pb-4">
            <div className="flex items-center gap-2">
              <Shield className="h-5 w-5 text-indigo-400" />
              <h2 className="text-base font-bold text-white">Administrator Access</h2>
            </div>
            <span className="inline-flex items-center gap-1 rounded-full bg-indigo-950/80 px-2.5 py-0.5 text-[11px] font-semibold text-indigo-300 border border-indigo-700/50">
              <Terminal className="h-3 w-3" />
              <span>Privileged Portal</span>
            </span>
          </div>

          {error && (
            <div className="rounded-2xl border border-rose-500/50 bg-rose-950/60 p-4 text-xs text-rose-200 flex items-start gap-3 shadow-xs">
              <AlertCircle className="h-4 w-4 shrink-0 text-rose-400 mt-0.5" />
              <div className="space-y-0.5">
                <strong className="font-semibold text-rose-100">Access Denied</strong>
                <p>{error}</p>
              </div>
            </div>
          )}

          <form onSubmit={handleSubmit} className="space-y-4">
            {/* Username Input */}
            <div className="space-y-1.5">
              <label className="block text-xs font-bold text-slate-300 uppercase tracking-wider">
                Admin Username
              </label>
              <div className="relative rounded-xl shadow-2xs">
                <div className="pointer-events-none absolute inset-y-0 left-0 flex items-center pl-3.5 text-slate-400">
                  <Lock className="h-4 w-4" />
                </div>
                <input
                  type="text"
                  required
                  autoFocus
                  autoComplete="username"
                  value={username}
                  onChange={(e) => setUsername(e.target.value)}
                  placeholder="e.g. admin"
                  className="block w-full rounded-xl border border-slate-600 bg-slate-900/80 pl-10 pr-3.5 py-2.5 text-sm font-medium text-white transition-all placeholder:text-slate-500 hover:border-slate-500 focus:border-indigo-400 focus:bg-slate-900 focus:outline-none focus:ring-4 focus:ring-indigo-500/20"
                />
              </div>
            </div>

            {/* Password Input */}
            <div className="space-y-1.5">
              <label className="block text-xs font-bold text-slate-300 uppercase tracking-wider">
                Admin Password
              </label>
              <div className="relative rounded-xl shadow-2xs">
                <div className="pointer-events-none absolute inset-y-0 left-0 flex items-center pl-3.5 text-slate-400">
                  <KeyRound className="h-4 w-4" />
                </div>
                <input
                  type={showPassword ? 'text' : 'password'}
                  required
                  autoComplete="current-password"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  placeholder="••••••••••••"
                  className="block w-full rounded-xl border border-slate-600 bg-slate-900/80 pl-10 pr-10 py-2.5 text-sm font-medium text-white transition-all placeholder:text-slate-500 hover:border-slate-500 focus:border-indigo-400 focus:bg-slate-900 focus:outline-none focus:ring-4 focus:ring-indigo-500/20"
                />
                <button
                  type="button"
                  onClick={() => setShowPassword(!showPassword)}
                  className="absolute inset-y-0 right-0 flex items-center pr-3 text-slate-400 hover:text-slate-200 cursor-pointer"
                >
                  {showPassword ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
                </button>
              </div>
            </div>

            <button
              type="submit"
              disabled={loading}
              className="inline-flex w-full items-center justify-center gap-2 rounded-xl bg-indigo-600 px-4 py-3 text-sm font-bold text-white shadow-lg shadow-indigo-600/30 transition-all hover:bg-indigo-500 hover:shadow-indigo-600/40 focus:outline-none focus:ring-4 focus:ring-indigo-500/30 disabled:cursor-not-allowed disabled:bg-slate-700 disabled:shadow-none cursor-pointer mt-2"
            >
              {loading ? (
                <>
                  <Sparkles className="h-4 w-4 animate-spin" />
                  <span>Authenticating Administrator...</span>
                </>
              ) : (
                <>
                  <span>Enter Admin Management</span>
                  <ArrowRight className="h-4 w-4" />
                </>
              )}
            </button>
          </form>

          <div className="rounded-2xl border border-slate-700/60 bg-slate-900/70 p-4 text-center space-y-1">
            <div className="flex items-center justify-center gap-1.5 text-amber-400 text-xs font-semibold">
              <ShieldAlert className="h-3.5 w-3.5" />
              <span>Privileged System Access</span>
            </div>
            <p className="text-[11px] text-slate-400 leading-relaxed">
              All administrative actions, account creations, and role modifications are logged for audit compliance.
            </p>
          </div>
        </div>

        <div className="mt-8 text-center text-xs text-slate-500">
          <span>CareGrid Administrator Console • Strictly Authorized Personnel Only</span>
        </div>
      </div>
    </div>
  )
}
