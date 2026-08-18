import React, { createContext, useContext, useState, useEffect } from 'react'
import { loginApi, logoutApi, getMeApi, setAuthToken } from '../services/api'

const AuthContext = createContext(null)

export const AuthProvider = ({ children }) => {
  const [user, setUser] = useState(null)
  const [token, setToken] = useState(null)
  const [loading, setLoading] = useState(true)

  // Initialize session on mount
  useEffect(() => {
    const initializeAuth = async () => {
      const storedToken = sessionStorage.getItem('caregrid_auth_token')
      if (storedToken) {
        setAuthToken(storedToken)
        setToken(storedToken)
        try {
          const profile = await getMeApi()
          setUser(profile)
        } catch (err) {
          console.warn('[Auth] Session expired or invalid:', err.message)
          sessionStorage.removeItem('caregrid_auth_token')
          setAuthToken(null)
          setToken(null)
          setUser(null)
        }
      }
      setLoading(false)
    }

    initializeAuth()

    const handleAuthExpired = () => {
      logout()
    }
    window.addEventListener('caregrid-auth-expired', handleAuthExpired)
    return () => window.removeEventListener('caregrid-auth-expired', handleAuthExpired)
  }, [])

  const login = async (username, password, expectedRole = null) => {
    const result = await loginApi(username, password)
    const { access_token, user: loggedUser } = result

    if (expectedRole && loggedUser.role !== expectedRole) {
      if (expectedRole === 'admin') {
        throw new Error('Access Denied: This console is strictly reserved for System Administrators.')
      } else if (expectedRole === 'nurse') {
        throw new Error('Please use your staff credentials to log in.')
      }
    }

    // Store token in session storage & memory
    sessionStorage.setItem('caregrid_auth_token', access_token)
    setAuthToken(access_token)
    setToken(access_token)
    setUser(loggedUser)
    return loggedUser
  }

  const logout = async () => {
    try {
      await logoutApi()
    } catch {
      // Best-effort logout
    } finally {
      sessionStorage.removeItem('caregrid_auth_token')
      setAuthToken(null)
      setToken(null)
      setUser(null)
    }
  }

  const value = {
    user,
    token,
    loading,
    login,
    logout,
    isAuthenticated: Boolean(user && token),
    isAdmin: user?.role === 'admin',
    isSubAdmin: user?.role === 'sub_admin',
    isStaffAdmin: user?.role === 'admin' || user?.role === 'sub_admin',
    isDoctor: user?.role === 'doctor',
    isNurse: user?.role === 'nurse',
  }

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>
}

export const useAuth = () => {
  const context = useContext(AuthContext)
  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider')
  }
  return context
}
