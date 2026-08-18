import axios from 'axios'

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 25000,
})

let currentAuthToken = null

export const setAuthToken = (token) => {
  currentAuthToken = token
  if (token) {
    api.defaults.headers.common['Authorization'] = `Bearer ${token}`
  } else {
    delete api.defaults.headers.common['Authorization']
  }
}

// Request Interceptor: Attach Bearer Token dynamically
api.interceptors.request.use(
  (config) => {
    if (currentAuthToken) {
      config.headers['Authorization'] = `Bearer ${currentAuthToken}`
    } else {
      // Check sessionStorage fallback if in-memory token isn't initialized yet
      const storedToken = sessionStorage.getItem('caregrid_auth_token')
      if (storedToken) {
        config.headers['Authorization'] = `Bearer ${storedToken}`
      }
    }
    return config
  },
  (error) => Promise.reject(error)
)

// Response Interceptor: Handle 401 Unauthorized / Token Expiry
api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      // Emit auth expired event for React context
      window.dispatchEvent(new CustomEvent('caregrid-auth-expired'))
    }
    return Promise.reject(error)
  }
)

// ── Auth Endpoints ──────────────────────────────────────────────────────────
export const loginApi = async (username, password) => {
  try {
    const response = await api.post('/auth/login', { username, password })
    return response.data
  } catch (error) {
    const detail = error.response?.data?.detail || 'Login failed. Please check your credentials.'
    throw new Error(detail, { cause: error })
  }
}

export const logoutApi = async () => {
  try {
    const response = await api.post('/auth/logout')
    return response.data
  } catch (error) {
    return { status: 'success' }
  }
}

export const getMeApi = async () => {
  try {
    const response = await api.get('/auth/me')
    return response.data
  } catch (error) {
    throw error
  }
}

// ── Admin Endpoints ─────────────────────────────────────────────────────────
export const adminGetUsersApi = async () => {
  try {
    const response = await api.get('/admin/users')
    return response.data
  } catch (error) {
    const detail = error.response?.data?.detail || 'Failed to retrieve staff accounts.'
    throw new Error(detail, { cause: error })
  }
}

export const adminCreateUserApi = async (userData) => {
  try {
    const response = await api.post('/admin/users', userData)
    return response.data
  } catch (error) {
    const detail = error.response?.data?.detail || 'Failed to create staff account.'
    throw new Error(detail, { cause: error })
  }
}

export const adminDeleteUserApi = async (userId) => {
  try {
    const response = await api.delete(`/admin/users/${userId}`)
    return response.data
  } catch (error) {
    const detail = error.response?.data?.detail || 'Failed to delete account.'
    throw new Error(detail, { cause: error })
  }
}

export const adminUpdatePasswordApi = async (userId, newPassword) => {
  try {
    const response = await api.patch(`/admin/users/${userId}/password`, { new_password: newPassword })
    return response.data
  } catch (error) {
    const detail = error.response?.data?.detail || 'Failed to update password.'
    throw new Error(detail, { cause: error })
  }
}

export const adminUpdateUserStatusApi = async (userId, isActive) => {
  try {
    const response = await api.patch(`/admin/users/${userId}/status`, { is_active: isActive })
    return response.data
  } catch (error) {
    const detail = error.response?.data?.detail || 'Failed to update account status.'
    throw new Error(detail, { cause: error })
  }
}

// ── ML Prediction & PDF Upload Endpoints ────────────────────────────────────
export const predictReadmission = async (patientData) => {
  try {
    const response = await api.post('/predict', patientData)
    return response.data
  } catch (error) {
    if (error.code === 'ERR_NETWORK' || error.message === 'Network Error') {
      throw new Error('Unable to connect to the prediction server. Please make sure the backend is running.', { cause: error })
    }

    if (error.response?.status === 401) {
      throw new Error('Session expired or unauthorized. Please log in again.', { cause: error })
    }

    if (error.response?.status === 400 || error.response?.status === 422) {
      throw new Error('Please review the patient details and try again.', { cause: error })
    }

    const detail = error.response?.data?.detail || 'Unable to compute prediction.'
    throw new Error(detail, { cause: error })
  }
}

export const uploadPdf = async (file) => {
  const formData = new FormData()
  formData.append('file', file)

  try {
    const response = await api.post('/upload-pdf', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    })
    return response.data
  } catch (error) {
    if (error.code === 'ERR_NETWORK' || error.message === 'Network Error') {
      throw new Error('Unable to connect to the PDF upload server. Please make sure the backend is running.', { cause: error })
    }

    if (error.response?.status === 401) {
      throw new Error('Session expired or unauthorized. Please log in again.', { cause: error })
    }

    const detail = error.response?.data?.detail || 'Unable to upload the PDF.'
    throw new Error(detail, { cause: error })
  }
}
