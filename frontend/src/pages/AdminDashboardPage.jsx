import React, { useState, useEffect } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { useAuth } from '../context/AuthContext'
import {
  adminGetUsersApi,
  adminCreateUserApi,
  adminUpdateUserStatusApi,
  adminUpdatePasswordApi,
  adminDeleteUserApi,
} from '../services/api'
import {
  Shield,
  UserPlus,
  Users,
  UserCheck,
  Stethoscope,
  LogOut,
  Sparkles,
  AlertCircle,
  CheckCircle2,
  Lock,
  User,
  KeyRound,
  Search,
  ArrowUpRight,
  ShieldCheck,
  Eye,
  EyeOff,
  Copy,
  Check,
  Edit3,
  Trash2,
  X,
  Briefcase,
  Sliders,
  CheckSquare,
  Square,
  ShieldAlert,
} from 'lucide-react'

export default function AdminDashboardPage() {
  const navigate = useNavigate()
  const { user: currentAdmin, logout, isAdmin } = useAuth()

  const [users, setUsers] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [successMsg, setSuccessMsg] = useState('')

  // Account creation form state
  const [newFullName, setNewFullName] = useState('')
  const [newDesignation, setNewDesignation] = useState('Staff Nurse')
  const [newUsername, setNewUsername] = useState('')
  const [newPassword, setNewPassword] = useState('')
  const [newRole, setNewRole] = useState('nurse')
  const [creating, setCreating] = useState(false)
  const [formError, setFormError] = useState('')

  // Sub-admin selective permissions
  const [permManageUsers, setPermManageUsers] = useState(true)
  const [permAuditReports, setPermAuditReports] = useState(true)

  // Password visibility map (userId -> bool)
  const [visiblePasswords, setVisiblePasswords] = useState({})
  const [copiedId, setCopiedId] = useState(null)

  // Edit Password Modal State
  const [editModalUser, setEditModalUser] = useState(null)
  const [editNewPassword, setEditNewPassword] = useState('')
  const [editLoading, setEditLoading] = useState(false)
  const [editError, setEditError] = useState('')

  // Delete Confirmation Modal State
  const [deleteModalUser, setDeleteModalUser] = useState(null)
  const [deleteLoading, setDeleteLoading] = useState(false)

  // Filter state
  const [searchQuery, setSearchQuery] = useState('')
  const [roleFilter, setRoleFilter] = useState('all')

  const fetchUsers = async () => {
    setLoading(true)
    setError('')
    try {
      const data = await adminGetUsersApi()
      setUsers(data)
    } catch (err) {
      setError(err.message || 'Failed to load staff accounts.')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchUsers()
  }, [])

  const togglePasswordVisibility = (userId) => {
    setVisiblePasswords((prev) => ({
      ...prev,
      [userId]: !prev[userId],
    }))
  }

  const handleCopyPassword = (userId, pwd) => {
    if (!pwd) return
    navigator.clipboard.writeText(pwd)
    setCopiedId(userId)
    setTimeout(() => setCopiedId(null), 2000)
  }

  const handleCreateUser = async (e) => {
    e.preventDefault()
    setFormError('')
    setSuccessMsg('')

    const trimmedUser = newUsername.trim().toLowerCase()
    const trimmedName = newFullName.trim()
    const trimmedDesignation = newDesignation.trim() || 'Staff'

    if (!trimmedUser || !trimmedName || !newPassword) {
      setFormError('Please fill in all required fields.')
      return
    }

    if (newPassword.length < 4) {
      setFormError('Password must be at least 4 characters long.')
      return
    }

    // Compile selective permissions for sub_admin or standard roles
    let permissionsStr = 'standard'
    if (newRole === 'sub_admin') {
      const perms = []
      if (permManageUsers) perms.push('create_users')
      if (permAuditReports) perms.push('audit_reports')
      permissionsStr = perms.join(',') || 'view_only'
    } else if (newRole === 'admin') {
      permissionsStr = 'all'
    }

    setCreating(true)
    try {
      const created = await adminCreateUserApi({
        username: trimmedUser,
        full_name: trimmedName,
        designation: trimmedDesignation,
        password: newPassword,
        role: newRole,
        permissions: permissionsStr,
      })

      setSuccessMsg(
        `Successfully created account for ${created.full_name} (${created.designation || 'Staff'}, @${created.username}) with password: ${newPassword}`
      )
      setNewFullName('')
      setNewDesignation(newRole === 'doctor' ? 'Doctor' : 'Staff Nurse')
      setNewUsername('')
      setNewPassword('')
      setNewRole('nurse')
      setPermManageUsers(true)
      setPermAuditReports(true)
      await fetchUsers()
    } catch (err) {
      setFormError(err.message || 'Failed to create user account.')
    } finally {
      setCreating(false)
    }
  }

  const handleOpenEditModal = (targetUser) => {
    setEditModalUser(targetUser)
    setEditNewPassword('')
    setEditError('')
  }

  const handleSavePassword = async (e) => {
    e.preventDefault()
    if (!editModalUser) return
    setEditError('')

    if (!editNewPassword || editNewPassword.length < 4) {
      setEditError('New password must be at least 4 characters long.')
      return
    }

    setEditLoading(true)
    try {
      await adminUpdatePasswordApi(editModalUser.id, editNewPassword)
      setSuccessMsg(`Password for @${editModalUser.username} has been updated to: ${editNewPassword}`)
      setEditModalUser(null)
      await fetchUsers()
    } catch (err) {
      setEditError(err.message || 'Failed to update password.')
    } finally {
      setEditLoading(false)
    }
  }

  const handleDeleteUser = async () => {
    if (!deleteModalUser) return
    setError('')
    setSuccessMsg('')

    if (deleteModalUser.id === currentAdmin?.id) {
      setError('Safety Protection: You cannot delete your own active administrator account.')
      setDeleteModalUser(null)
      return
    }

    setDeleteLoading(true)
    try {
      await adminDeleteUserApi(deleteModalUser.id)
      setSuccessMsg(`User @${deleteModalUser.username} has been permanently deleted.`)
      setDeleteModalUser(null)
      await fetchUsers()
    } catch (err) {
      setError(err.message || 'Failed to delete user account.')
    } finally {
      setDeleteLoading(false)
    }
  }

  const handleToggleStatus = async (targetUser) => {
    setError('')
    setSuccessMsg('')

    if (targetUser.id === currentAdmin?.id) {
      setError('Safety Protection: You cannot deactivate your own active administrator account.')
      return
    }

    try {
      const newStatus = !targetUser.is_active
      await adminUpdateUserStatusApi(targetUser.id, newStatus)
      setSuccessMsg(
        `Account @${targetUser.username} has been ${newStatus ? 'reactivated' : 'deactivated'}.`
      )
      await fetchUsers()
    } catch (err) {
      setError(err.message || 'Failed to update account status.')
    }
  }

  const handleLogout = async () => {
    await logout()
    navigate('/login', { replace: true })
  }

  // Filtered users list
  const filteredUsers = users.filter((u) => {
    const matchesSearch =
      u.full_name.toLowerCase().includes(searchQuery.toLowerCase()) ||
      u.username.toLowerCase().includes(searchQuery.toLowerCase()) ||
      (u.designation && u.designation.toLowerCase().includes(searchQuery.toLowerCase()))
    const matchesRole = roleFilter === 'all' || u.role === roleFilter
    return matchesSearch && matchesRole
  })

  const totalUsers = users.length
  const totalNurses = users.filter((u) => u.role === 'nurse' && u.is_active).length
  const totalDoctors = users.filter((u) => u.role === 'doctor' && u.is_active).length
  const totalSubAdmins = users.filter((u) => u.role === 'sub_admin' && u.is_active).length
  const totalAdmins = users.filter((u) => u.role === 'admin' && u.is_active).length

  return (
    <div className="min-h-screen bg-slate-50 text-slate-900 flex flex-col selection:bg-indigo-600 selection:text-white">
      {/* ─────────────────────────────────────────────────────────────
          1. ADMIN CONSOLE HEADER
      ───────────────────────────────────────────────────────────── */}
      <header className="sticky top-0 z-30 border-b border-slate-200 bg-white/95 backdrop-blur-sm shadow-2xs">
        <div className="mx-auto max-w-7xl px-4 py-3 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-indigo-50 border border-indigo-100 text-indigo-600 shadow-xs">
                <Shield className="h-5 w-5" />
              </div>
              <div>
                <div className="flex items-center gap-2">
                  <h1 className="text-lg sm:text-xl font-bold tracking-tight text-slate-900">
                    CareGrid Admin Console
                  </h1>
                  <span className="rounded-md bg-indigo-100 px-2 py-0.5 text-[10px] font-extrabold uppercase tracking-wider text-indigo-800">
                    {currentAdmin?.role === 'sub_admin' ? 'Sub-Admin' : 'Super Admin'}
                  </span>
                </div>
                <p className="text-xs text-slate-500">
                  Staff Provisioning, Sub-Admin Roles, Designations & Credentials
                </p>
              </div>
            </div>

            <div className="flex items-center gap-2 sm:gap-3">
              <Link
                to="/"
                className="inline-flex items-center gap-1.5 rounded-xl border border-slate-200 bg-white px-3.5 py-2 text-xs font-bold text-slate-700 shadow-xs transition hover:border-slate-300 hover:bg-slate-50 cursor-pointer"
              >
                <Stethoscope className="h-4 w-4 text-blue-600" />
                <span className="hidden sm:inline">Clinical Prediction Dashboard</span>
                <span className="sm:hidden">Clinical</span>
                <ArrowUpRight className="h-3.5 w-3.5 text-slate-400" />
              </Link>

              <button
                type="button"
                onClick={handleLogout}
                className="inline-flex items-center gap-1.5 rounded-xl border border-rose-200 bg-rose-50/60 px-3.5 py-2 text-xs font-bold text-rose-700 transition hover:bg-rose-100 cursor-pointer"
              >
                <LogOut className="h-4 w-4" />
                <span className="hidden sm:inline">Sign Out</span>
              </button>
            </div>
          </div>
        </div>
      </header>

      {/* ─────────────────────────────────────────────────────────────
          2. MAIN CONTENT
      ───────────────────────────────────────────────────────────── */}
      <main className="flex-1 mx-auto w-full max-w-7xl px-4 py-6 sm:px-6 lg:px-8 space-y-6">
        {/* Alerts */}
        {successMsg && (
          <div className="rounded-2xl border border-emerald-200 bg-emerald-50 p-4 text-xs font-semibold text-emerald-900 flex items-center justify-between shadow-xs">
            <div className="flex items-center gap-2.5">
              <CheckCircle2 className="h-4 w-4 text-emerald-600 shrink-0" />
              <span>{successMsg}</span>
            </div>
            <button
              onClick={() => setSuccessMsg('')}
              className="text-emerald-700 hover:text-emerald-900 cursor-pointer text-xs font-bold"
            >
              Dismiss
            </button>
          </div>
        )}

        {error && (
          <div className="rounded-2xl border border-rose-200 bg-rose-50 p-4 text-xs font-semibold text-rose-900 flex items-center justify-between shadow-xs">
            <div className="flex items-center gap-2.5">
              <AlertCircle className="h-4 w-4 text-rose-600 shrink-0" />
              <span>{error}</span>
            </div>
            <button
              onClick={() => setError('')}
              className="text-rose-700 hover:text-rose-900 cursor-pointer text-xs font-bold"
            >
              Dismiss
            </button>
          </div>
        )}

        {/* Overview Stat Cards */}
        <div className="grid gap-4 grid-cols-2 sm:grid-cols-4">
          <div className="rounded-2xl border border-slate-200 bg-white p-4 shadow-xs flex items-center justify-between">
            <div>
              <p className="text-[11px] font-bold text-slate-500 uppercase tracking-wider">Total Accounts</p>
              <h3 className="mt-1 text-2xl font-black text-slate-900">{totalUsers}</h3>
              <span className="text-[10px] text-slate-500">All registered users</span>
            </div>
            <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-slate-100 text-slate-700">
              <Users className="h-5 w-5" />
            </div>
          </div>

          <div className="rounded-2xl border border-blue-100 bg-blue-50/40 p-4 shadow-xs flex items-center justify-between">
            <div>
              <p className="text-[11px] font-bold text-blue-700 uppercase tracking-wider">Nurses & Doctors</p>
              <h3 className="mt-1 text-2xl font-black text-blue-900">{totalNurses + totalDoctors}</h3>
              <span className="text-[10px] text-blue-600">{totalDoctors} Doctors • {totalNurses} Nurses</span>
            </div>
            <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-blue-100 text-blue-700">
              <UserCheck className="h-5 w-5" />
            </div>
          </div>

          <div className="rounded-2xl border border-indigo-100 bg-indigo-50/40 p-4 shadow-xs flex items-center justify-between">
            <div>
              <p className="text-[11px] font-bold text-indigo-700 uppercase tracking-wider">Sub-Admins</p>
              <h3 className="mt-1 text-2xl font-black text-indigo-900">{totalSubAdmins}</h3>
              <span className="text-[10px] text-indigo-600">With staff intake roles</span>
            </div>
            <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-indigo-100 text-indigo-700">
              <Shield className="h-5 w-5" />
            </div>
          </div>

          <div className="rounded-2xl border border-purple-100 bg-purple-50/40 p-4 shadow-xs flex items-center justify-between">
            <div>
              <p className="text-[11px] font-bold text-purple-700 uppercase tracking-wider">Super Admins</p>
              <h3 className="mt-1 text-2xl font-black text-purple-900">{totalAdmins}</h3>
              <span className="text-[10px] text-purple-600">Full system access</span>
            </div>
            <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-purple-100 text-purple-700">
              <ShieldCheck className="h-5 w-5" />
            </div>
          </div>
        </div>

        <div className="grid gap-6 lg:grid-cols-12">
          {/* Form: Provision New Account (Sub-Admin / Doctor / Nurse) */}
          <div className="lg:col-span-5 rounded-2xl border border-slate-200 bg-white p-6 shadow-xs space-y-5 h-fit">
            <div className="flex items-center gap-2.5 border-b border-slate-100 pb-3">
              <UserPlus className="h-5 w-5 text-indigo-600" />
              <div>
                <h2 className="text-base font-bold text-slate-900">Provision New Staff / Sub-Admin</h2>
                <p className="text-[11px] text-slate-500">Create login credentials with designation and selective roles</p>
              </div>
            </div>

            {formError && (
              <div className="rounded-xl border border-rose-200 bg-rose-50 p-3 text-xs text-rose-800 flex items-start gap-2">
                <AlertCircle className="h-4 w-4 shrink-0 text-rose-600 mt-0.5" />
                <span>{formError}</span>
              </div>
            )}

            <form onSubmit={handleCreateUser} className="space-y-4">
              {/* Role Selection */}
              <div className="space-y-1">
                <label className="block text-xs font-bold text-slate-700 uppercase tracking-wider">
                  Select System Role
                </label>
                <select
                  value={newRole}
                  onChange={(e) => {
                    const r = e.target.value
                    setNewRole(r)
                    if (r === 'doctor') setNewDesignation('Doctor / Physician')
                    else if (r === 'sub_admin') setNewDesignation('Department Lead / Doctor')
                    else if (r === 'admin') setNewDesignation('System Administrator')
                    else setNewDesignation('Staff Nurse')
                  }}
                  className="block w-full rounded-xl border border-slate-200 bg-slate-50 px-3 py-2 text-xs font-bold text-slate-900 hover:border-slate-300 focus:border-indigo-500 focus:bg-white focus:outline-none focus:ring-2 focus:ring-indigo-500/10 cursor-pointer"
                >
                  <option value="nurse">Clinical Nurse (Standard User)</option>
                  <option value="doctor">Medical Doctor / Physician (Standard User)</option>
                  <option value="sub_admin">Sub-Admin (Selective Roles: Add New Users & Audit)</option>
                  {isAdmin && <option value="admin">Super Admin (Full System Access)</option>}
                </select>
              </div>

              {/* Designation / Job Title Input with Quick Presets */}
              <div className="space-y-1">
                <div className="flex items-center justify-between">
                  <label className="block text-xs font-bold text-slate-700 uppercase tracking-wider">
                    Designation / Title
                  </label>
                  <span className="text-[10px] text-slate-400 font-semibold">Editable Job Title</span>
                </div>
                <div className="relative">
                  <div className="pointer-events-none absolute inset-y-0 left-0 flex items-center pl-3 text-slate-400">
                    <Briefcase className="h-4 w-4" />
                  </div>
                  <input
                    type="text"
                    required
                    value={newDesignation}
                    onChange={(e) => setNewDesignation(e.target.value)}
                    placeholder="e.g. Doctor, Lead Cardiologist, Charge Nurse"
                    className="block w-full rounded-xl border border-slate-200 bg-slate-50 pl-9 pr-3 py-2 text-xs font-semibold text-slate-900 hover:border-slate-300 focus:border-indigo-500 focus:bg-white focus:outline-none focus:ring-2 focus:ring-indigo-500/10"
                  />
                </div>
                {/* Quick Presets */}
                <div className="flex flex-wrap gap-1.5 pt-1">
                  {['Doctor', 'Staff Nurse', 'Charge Nurse', 'Clinical Supervisor', 'Cardiology Lead'].map((preset) => (
                    <button
                      key={preset}
                      type="button"
                      onClick={() => setNewDesignation(preset)}
                      className="rounded-lg border border-slate-200 bg-slate-100/80 px-2 py-0.5 text-[10px] font-semibold text-slate-600 hover:bg-slate-200 hover:text-slate-900 cursor-pointer"
                    >
                      {preset}
                    </button>
                  ))}
                </div>
              </div>

              {/* Full Name */}
              <div className="space-y-1">
                <label className="block text-xs font-bold text-slate-700 uppercase tracking-wider">
                  Full Name
                </label>
                <div className="relative">
                  <div className="pointer-events-none absolute inset-y-0 left-0 flex items-center pl-3 text-slate-400">
                    <User className="h-4 w-4" />
                  </div>
                  <input
                    type="text"
                    required
                    value={newFullName}
                    onChange={(e) => setNewFullName(e.target.value)}
                    placeholder="e.g. Dr. Robert Chen, MD or Sarah Connor, RN"
                    className="block w-full rounded-xl border border-slate-200 bg-slate-50 pl-9 pr-3 py-2 text-xs font-medium text-slate-900 hover:border-slate-300 focus:border-indigo-500 focus:bg-white focus:outline-none focus:ring-2 focus:ring-indigo-500/10"
                  />
                </div>
              </div>

              {/* Username */}
              <div className="space-y-1">
                <label className="block text-xs font-bold text-slate-700 uppercase tracking-wider">
                  Username
                </label>
                <div className="relative">
                  <div className="pointer-events-none absolute inset-y-0 left-0 flex items-center pl-3 text-slate-400">
                    <span className="text-xs font-bold text-slate-400">@</span>
                  </div>
                  <input
                    type="text"
                    required
                    value={newUsername}
                    onChange={(e) => setNewUsername(e.target.value)}
                    placeholder="e.g. dr_chen or nurse_sarah"
                    className="block w-full rounded-xl border border-slate-200 bg-slate-50 pl-8 pr-3 py-2 text-xs font-medium text-slate-900 hover:border-slate-300 focus:border-indigo-500 focus:bg-white focus:outline-none focus:ring-2 focus:ring-indigo-500/10 font-mono"
                  />
                </div>
              </div>

              {/* Assigned Password */}
              <div className="space-y-1">
                <label className="block text-xs font-bold text-slate-700 uppercase tracking-wider">
                  Assigned Password
                </label>
                <div className="relative">
                  <div className="pointer-events-none absolute inset-y-0 left-0 flex items-center pl-3 text-slate-400">
                    <KeyRound className="h-4 w-4" />
                  </div>
                  <input
                    type="text"
                    required
                    value={newPassword}
                    onChange={(e) => setNewPassword(e.target.value)}
                    placeholder="e.g. Pass@1234"
                    className="block w-full rounded-xl border border-slate-200 bg-slate-50 pl-9 pr-3 py-2 text-xs font-medium text-slate-900 hover:border-slate-300 focus:border-indigo-500 focus:bg-white focus:outline-none focus:ring-2 focus:ring-indigo-500/10 font-mono"
                  />
                </div>
              </div>

              {/* Selective Roles / Permissions Box for Sub-Admin */}
              {newRole === 'sub_admin' && (
                <div className="rounded-xl border border-indigo-200 bg-indigo-50/50 p-3.5 space-y-2.5">
                  <div className="flex items-center gap-1.5 text-indigo-900 font-bold text-xs">
                    <Sliders className="h-4 w-4 text-indigo-600" />
                    <span>Selective Sub-Admin Permissions</span>
                  </div>
                  <p className="text-[11px] text-indigo-700">
                    Configure which administrative capabilities this Sub-Admin ({newDesignation || 'Doctor'}) can execute:
                  </p>

                  <div className="space-y-2 pt-1">
                    <label className="flex items-start gap-2 cursor-pointer select-none">
                      <input
                        type="checkbox"
                        checked={permManageUsers}
                        onChange={(e) => setPermManageUsers(e.target.checked)}
                        className="mt-0.5 rounded text-indigo-600 focus:ring-indigo-500"
                      />
                      <div className="text-[11px] leading-snug">
                        <strong className="text-slate-900">Add & Manage Staff (Doctors/Nurses)</strong>
                        <p className="text-slate-500">Can provision new clinical accounts and view staff directory.</p>
                      </div>
                    </label>

                    <label className="flex items-start gap-2 cursor-pointer select-none">
                      <input
                        type="checkbox"
                        checked={permAuditReports}
                        onChange={(e) => setPermAuditReports(e.target.checked)}
                        className="mt-0.5 rounded text-indigo-600 focus:ring-indigo-500"
                      />
                      <div className="text-[11px] leading-snug">
                        <strong className="text-slate-900">Clinical Audit & System Oversight</strong>
                        <p className="text-slate-500">Can review prediction history, high-risk logs, and patient analytics.</p>
                      </div>
                    </label>
                  </div>
                </div>
              )}

              <button
                type="submit"
                disabled={creating}
                className="inline-flex w-full items-center justify-center gap-2 rounded-xl bg-indigo-600 px-4 py-2.5 text-xs font-bold text-white shadow-xs transition hover:bg-indigo-700 disabled:cursor-not-allowed disabled:bg-slate-300 cursor-pointer"
              >
                {creating ? (
                  <>
                    <Sparkles className="h-3.5 w-3.5 animate-spin" />
                    <span>Provisioning Account...</span>
                  </>
                ) : (
                  <>
                    <UserPlus className="h-3.5 w-3.5" />
                    <span>Provision Account</span>
                  </>
                )}
              </button>
            </form>
          </div>

          {/* Directory: List All Accounts */}
          <div className="lg:col-span-7 rounded-2xl border border-slate-200 bg-white p-6 shadow-xs space-y-4">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-slate-100 pb-3">
              <div>
                <h2 className="text-base font-bold text-slate-900">Staff & Account Directory</h2>
                <p className="text-[11px] text-slate-500">View designations, selective roles, passwords, and manage accounts</p>
              </div>

              {/* Filters */}
              <div className="flex items-center gap-2">
                <div className="relative">
                  <Search className="pointer-events-none absolute inset-y-0 left-0 pl-2.5 h-3.5 w-3.5 my-auto text-slate-400" />
                  <input
                    type="text"
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                    placeholder="Search name, title, user..."
                    className="rounded-xl border border-slate-200 bg-slate-50 pl-8 pr-3 py-1.5 text-xs text-slate-800 placeholder:text-slate-400 focus:outline-none focus:ring-2 focus:ring-indigo-500/10"
                  />
                </div>

                <select
                  value={roleFilter}
                  onChange={(e) => setRoleFilter(e.target.value)}
                  className="rounded-xl border border-slate-200 bg-slate-50 px-2.5 py-1.5 text-xs font-medium text-slate-700 focus:outline-none focus:ring-2 focus:ring-indigo-500/10 cursor-pointer"
                >
                  <option value="all">All Roles</option>
                  <option value="nurse">Nurses</option>
                  <option value="doctor">Doctors</option>
                  <option value="sub_admin">Sub-Admins</option>
                  <option value="admin">Super Admins</option>
                </select>
              </div>
            </div>

            {loading ? (
              <div className="py-12 text-center text-xs text-slate-500">
                <Sparkles className="h-6 w-6 animate-spin mx-auto text-indigo-600 mb-2" />
                <span>Loading staff directory...</span>
              </div>
            ) : filteredUsers.length === 0 ? (
              <div className="py-12 text-center text-xs text-slate-500">
                <Users className="h-8 w-8 mx-auto text-slate-300 mb-2" />
                <p className="font-semibold text-slate-700">No matching accounts found.</p>
                <p className="mt-0.5">Try adjusting your search query or role filter.</p>
              </div>
            ) : (
              <div className="overflow-x-auto">
                <table className="w-full text-left text-xs text-slate-700">
                  <thead className="border-b border-slate-100 bg-slate-50/70 text-[11px] font-bold uppercase tracking-wider text-slate-500">
                    <tr>
                      <th className="py-2.5 px-3">User & Designation</th>
                      <th className="py-2.5 px-3">Role</th>
                      <th className="py-2.5 px-3">Password (Admin View)</th>
                      <th className="py-2.5 px-3">Status</th>
                      <th className="py-2.5 px-3 text-right">Actions</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-100">
                    {filteredUsers.map((u) => {
                      const isSelf = u.id === currentAdmin?.id
                      const isVisible = visiblePasswords[u.id] || false
                      const displayPwd = u.initial_password || (u.username === 'admin' ? 'Admin@123' : '••••••••')

                      return (
                        <tr key={u.id} className="hover:bg-slate-50/60 transition-colors">
                          <td className="py-3 px-3">
                            <div className="font-bold text-slate-900 flex items-center gap-1.5">
                              <span>{u.full_name}</span>
                            </div>
                            <div className="flex items-center gap-2 mt-0.5">
                              <span className="inline-flex items-center rounded-md bg-slate-100 px-1.5 py-0.5 text-[10px] font-semibold text-slate-700 border border-slate-200">
                                {u.designation || 'Staff'}
                              </span>
                              <span className="text-[11px] font-mono text-slate-400">@{u.username}</span>
                            </div>
                          </td>

                          {/* Role Badge */}
                          <td className="py-3 px-3">
                            {u.role === 'admin' ? (
                              <span className="inline-flex items-center gap-1 rounded-md bg-purple-50 px-2 py-0.5 text-[10px] font-extrabold text-purple-700 border border-purple-100">
                                <ShieldCheck className="h-3 w-3" />
                                <span>Super Admin</span>
                              </span>
                            ) : u.role === 'sub_admin' ? (
                              <div className="space-y-1">
                                <span className="inline-flex items-center gap-1 rounded-md bg-indigo-50 px-2 py-0.5 text-[10px] font-extrabold text-indigo-700 border border-indigo-100">
                                  <Shield className="h-3 w-3" />
                                  <span>Sub-Admin</span>
                                </span>
                                <div className="text-[9px] text-indigo-600 font-semibold">
                                  {u.permissions?.includes('create_users') ? '✓ Can Add Users' : 'Intake Only'}
                                </div>
                              </div>
                            ) : u.role === 'doctor' ? (
                              <span className="inline-flex items-center gap-1 rounded-md bg-teal-50 px-2 py-0.5 text-[10px] font-bold text-teal-700 border border-teal-100">
                                <Stethoscope className="h-3 w-3" />
                                <span>Doctor</span>
                              </span>
                            ) : (
                              <span className="inline-flex items-center gap-1 rounded-md bg-blue-50 px-2 py-0.5 text-[10px] font-bold text-blue-700 border border-blue-100">
                                <User className="h-3 w-3" />
                                <span>Nurse</span>
                              </span>
                            )}
                          </td>

                          {/* Password Column with View & Copy */}
                          <td className="py-3 px-3">
                            <div className="flex items-center gap-1.5">
                              <span className="font-mono text-xs text-slate-900 bg-slate-100 px-2 py-0.5 rounded border border-slate-200">
                                {isVisible ? displayPwd : '••••••••'}
                              </span>

                              <button
                                type="button"
                                onClick={() => togglePasswordVisibility(u.id)}
                                title={isVisible ? 'Hide Password' : 'View Password'}
                                className="p-1 rounded text-slate-400 hover:text-slate-700 hover:bg-slate-100 cursor-pointer"
                              >
                                {isVisible ? <EyeOff className="h-3.5 w-3.5" /> : <Eye className="h-3.5 w-3.5" />}
                              </button>

                              {displayPwd && displayPwd !== '••••••••' && (
                                <button
                                  type="button"
                                  onClick={() => handleCopyPassword(u.id, displayPwd)}
                                  title="Copy Password"
                                  className="p-1 rounded text-slate-400 hover:text-indigo-600 hover:bg-slate-100 cursor-pointer"
                                >
                                  {copiedId === u.id ? (
                                    <Check className="h-3.5 w-3.5 text-emerald-600" />
                                  ) : (
                                    <Copy className="h-3.5 w-3.5" />
                                  )}
                                </button>
                              )}
                            </div>
                          </td>

                          {/* Status */}
                          <td className="py-3 px-3">
                            {u.is_active ? (
                              <span className="inline-flex items-center gap-1 rounded-full bg-emerald-50 px-2 py-0.5 text-[10px] font-semibold text-emerald-700 border border-emerald-100">
                                <span className="h-1.5 w-1.5 rounded-full bg-emerald-600"></span>
                                <span>Active</span>
                              </span>
                            ) : (
                              <span className="inline-flex items-center gap-1 rounded-full bg-rose-50 px-2 py-0.5 text-[10px] font-semibold text-rose-700 border border-rose-100">
                                <span className="h-1.5 w-1.5 rounded-full bg-rose-600"></span>
                                <span>Deactivated</span>
                              </span>
                            )}
                          </td>

                          {/* Actions */}
                          <td className="py-3 px-3 text-right">
                            <div className="flex items-center justify-end gap-1">
                              {/* Edit Password Button */}
                              <button
                                type="button"
                                onClick={() => handleOpenEditModal(u)}
                                title="Edit / Reset Password"
                                className="inline-flex items-center gap-1 rounded-lg border border-slate-200 bg-white px-2 py-1 text-[11px] font-bold text-slate-700 hover:bg-slate-50 hover:border-slate-300 cursor-pointer"
                              >
                                <Edit3 className="h-3 w-3 text-blue-600" />
                                <span>Edit Pass</span>
                              </button>

                              {!isSelf && (
                                <>
                                  {/* Toggle Status */}
                                  <button
                                    type="button"
                                    onClick={() => handleToggleStatus(u)}
                                    title={u.is_active ? 'Deactivate User' : 'Reactivate User'}
                                    className={`inline-flex items-center gap-1 rounded-lg px-2 py-1 text-[11px] font-bold transition cursor-pointer ${
                                      u.is_active
                                        ? 'border border-amber-200 bg-white text-amber-700 hover:bg-amber-50'
                                        : 'border border-emerald-200 bg-white text-emerald-700 hover:bg-emerald-50'
                                    }`}
                                  >
                                    {u.is_active ? 'Deactivate' : 'Reactivate'}
                                  </button>

                                  {/* Delete User Button */}
                                  <button
                                    type="button"
                                    onClick={() => setDeleteModalUser(u)}
                                    title="Delete User Permanently"
                                    className="inline-flex items-center gap-1 rounded-lg border border-rose-200 bg-white px-2 py-1 text-[11px] font-bold text-rose-600 hover:bg-rose-50 cursor-pointer"
                                  >
                                    <Trash2 className="h-3 w-3" />
                                    <span>Delete</span>
                                  </button>
                                </>
                              )}

                              {isSelf && (
                                <span className="text-[10px] font-semibold text-slate-400 italic px-1">
                                  Current User
                                </span>
                              )}
                            </div>
                          </td>
                        </tr>
                      )
                    })}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        </div>
      </main>

      {/* ─────────────────────────────────────────────────────────────
          3. EDIT PASSWORD MODAL
      ───────────────────────────────────────────────────────────── */}
      {editModalUser && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-900/50 backdrop-blur-xs p-4">
          <div className="w-full max-w-md rounded-2xl border border-slate-200 bg-white p-6 shadow-2xl space-y-4 animate-in fade-in zoom-in-95">
            <div className="flex items-center justify-between border-b border-slate-100 pb-3">
              <div className="flex items-center gap-2">
                <KeyRound className="h-5 w-5 text-indigo-600" />
                <h3 className="text-base font-bold text-slate-900">
                  Edit Password for @{editModalUser.username}
                </h3>
              </div>
              <button
                type="button"
                onClick={() => setEditModalUser(null)}
                className="rounded-lg p-1 text-slate-400 hover:bg-slate-100 cursor-pointer"
              >
                <X className="h-4 w-4" />
              </button>
            </div>

            {editError && (
              <div className="rounded-xl border border-rose-200 bg-rose-50 p-3 text-xs text-rose-800 flex items-start gap-2">
                <AlertCircle className="h-4 w-4 shrink-0 text-rose-600 mt-0.5" />
                <span>{editError}</span>
              </div>
            )}

            <form onSubmit={handleSavePassword} className="space-y-4">
              <div className="space-y-1">
                <label className="block text-xs font-bold text-slate-700 uppercase tracking-wider">
                  New Password
                </label>
                <input
                  type="text"
                  required
                  autoFocus
                  value={editNewPassword}
                  onChange={(e) => setEditNewPassword(e.target.value)}
                  placeholder="Enter new password (min 4 characters)"
                  className="block w-full rounded-xl border border-slate-200 bg-slate-50 px-3 py-2 text-sm font-mono text-slate-900 hover:border-slate-300 focus:border-indigo-500 focus:bg-white focus:outline-none focus:ring-2 focus:ring-indigo-500/10"
                />
                <p className="text-[11px] text-slate-500">
                  The password will be hashed with Bcrypt and visible to administrators in the directory.
                </p>
              </div>

              <div className="flex items-center justify-end gap-2 pt-2 border-t border-slate-100">
                <button
                  type="button"
                  onClick={() => setEditModalUser(null)}
                  className="rounded-xl border border-slate-200 bg-white px-4 py-2 text-xs font-bold text-slate-700 hover:bg-slate-50 cursor-pointer"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  disabled={editLoading}
                  className="inline-flex items-center gap-1.5 rounded-xl bg-indigo-600 px-4 py-2 text-xs font-bold text-white hover:bg-indigo-700 disabled:bg-slate-300 cursor-pointer shadow-xs"
                >
                  {editLoading ? <Sparkles className="h-3.5 w-3.5 animate-spin" /> : null}
                  <span>Save New Password</span>
                </button>
              </div>
            </form>
          </div>
        </div>
      )}

      {/* ─────────────────────────────────────────────────────────────
          4. DELETE CONFIRMATION MODAL
      ───────────────────────────────────────────────────────────── */}
      {deleteModalUser && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-900/50 backdrop-blur-xs p-4">
          <div className="w-full max-w-md rounded-2xl border border-slate-200 bg-white p-6 shadow-2xl space-y-4 animate-in fade-in zoom-in-95">
            <div className="flex items-center gap-3 text-rose-600">
              <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-rose-50 border border-rose-100">
                <Trash2 className="h-5 w-5 text-rose-600" />
              </div>
              <div>
                <h3 className="text-base font-bold text-slate-900">Confirm Account Deletion</h3>
                <p className="text-xs text-slate-500">This action cannot be undone</p>
              </div>
            </div>

            <p className="text-xs text-slate-600 leading-relaxed">
              Are you sure you want to permanently delete the account for{' '}
              <strong className="text-slate-900">{deleteModalUser.full_name}</strong> (
              <span className="font-mono text-slate-700">@{deleteModalUser.username}</span>,{' '}
              <span className="font-semibold text-slate-800">{deleteModalUser.designation || 'Staff'}</span>)?
            </p>

            <div className="flex items-center justify-end gap-2 pt-2 border-t border-slate-100">
              <button
                type="button"
                onClick={() => setDeleteModalUser(null)}
                className="rounded-xl border border-slate-200 bg-white px-4 py-2 text-xs font-bold text-slate-700 hover:bg-slate-50 cursor-pointer"
              >
                Cancel
              </button>
              <button
                type="button"
                onClick={handleDeleteUser}
                disabled={deleteLoading}
                className="inline-flex items-center gap-1.5 rounded-xl bg-rose-600 px-4 py-2 text-xs font-bold text-white hover:bg-rose-700 disabled:bg-slate-300 cursor-pointer shadow-xs"
              >
                {deleteLoading ? <Sparkles className="h-3.5 w-3.5 animate-spin" /> : null}
                <span>Delete Account Permanently</span>
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
