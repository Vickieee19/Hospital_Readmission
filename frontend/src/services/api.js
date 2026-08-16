import axios from 'axios'

const api = axios.create({
  baseURL: import.meta.env.VITE_API_URL || 'http://localhost:8000',
  timeout: 20000,
})

export const predictReadmission = async (patientData) => {
  try {
    const response = await api.post('/predict', patientData)
    return response.data
  } catch (error) {
    if (error.code === 'ERR_NETWORK' || error.message === 'Network Error') {
      throw new Error('Unable to connect to the prediction server. Please make sure the backend is running.')
    }

    if (error.response?.status === 400 || error.response?.status === 422) {
      throw new Error('Please review the patient details and try again.')
    }

    throw new Error('Unable to connect to the prediction server. Please make sure the backend is running.')
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
      throw new Error('Unable to connect to the PDF upload server. Please make sure the backend is running.')
    }

    const detail = error.response?.data?.detail || 'Unable to upload the PDF.'
    throw new Error(detail)
  }
}
