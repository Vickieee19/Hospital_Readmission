export const AGE_OPTIONS = ['[40-50)', '[50-60)', '[60-70)', '[70-80)', '[80-90)', '[90-100)']
export const SPECIALTIES = [
  'InternalMedicine',
  'Family/GeneralPractice',
  'Cardiology',
  'Surgery',
  'Other',
  'Missing',
]
export const DIAG_OPTIONS = ['Other', 'Circulatory', 'Diabetes', 'Respiratory', 'Digestive', 'Injury']
export const BINARY_OPTIONS = ['no', 'yes']
export const GLUCOSE_OPTIONS = ['no', 'normal', 'high', 'Missing']
export const A1C_OPTIONS = ['no', 'normal', 'high', 'Missing']

export const defaultValues = {
  age: '[70-80)',
  time_in_hospital: 5,
  n_lab_procedures: 40,
  n_procedures: 2,
  n_medications: 15,
  n_outpatient: 0,
  n_inpatient: 1,
  n_emergency: 0,
  medical_specialty: 'InternalMedicine',
  diag_1: 'Circulatory',
  diag_2: 'Diabetes',
  diag_3: 'Other',
  glucose_test: 'no',
  A1Ctest: 'no',
  change: 'yes',
  diabetes_med: 'yes',
  threshold: 0.52,
}

