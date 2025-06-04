import { useState } from 'react'
import Head from 'next/head'
import { useForm, SubmitHandler } from 'react-hook-form'
import axios from 'axios'

// Types
type RecipientInfo = {
  name: string
  email: string
  company?: string
  role?: string
  linkedin_url?: string
}

type EmailFormInput = {
  recipient: RecipientInfo
  goal: string
  additional_context?: string
  tone: string
  use_lead_scoring: boolean
}

type EmailResponse = {
  email_id: string
  subject: string
  body: string
  recipient_email: string
  personalization_data: Record<string, any>
  lead_score?: number
  created_at: string
}

// Main Page Component
export default function Home() {
  const [isLoading, setIsLoading] = useState(false)
  const [emailDraft, setEmailDraft] = useState<EmailResponse | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [pushStatus, setPushStatus] = useState<'idle' | 'success' | 'error'>('idle')

  const { register, handleSubmit, formState: { errors } } = useForm<EmailFormInput>({
    defaultValues: {
      recipient: {
        name: '',
        email: '',
        company: '',
        role: '',
        linkedin_url: ''
      },
      goal: '',
      additional_context: '',
      tone: 'professional',
      use_lead_scoring: true
    }
  })

  const onSubmit: SubmitHandler<EmailFormInput> = async (data: EmailFormInput) => {
    setIsLoading(true)
    setError(null)
    
    try {
      // In production, this would point to your actual API
      const response = await axios.post<EmailResponse>('/api/email/generate', data)
      setEmailDraft(response.data)
    } catch (err) {
      console.error('Error generating email:', err)
      setError('Failed to generate email. Please try again.')
    } finally {
      setIsLoading(false)
    }
  }

  const handlePushToHubspot = async () => {
    if (!emailDraft) return
    
    try {
      setPushStatus('idle')
      // In production, this would point to your actual API
      await axios.post('/api/email/push-to-hubspot', {
        email_id: emailDraft.email_id
      })
      setPushStatus('success')
    } catch (err) {
      console.error('Error pushing to HubSpot:', err)
      setPushStatus('error')
    }
  }

  return (
    <div className="min-h-screen bg-gray-50">
      <Head>
        <title>ColdEmailIO</title>
        <meta name="description" content="AI-powered cold email personalization" />
        <link rel="icon" href="/favicon.ico" />
      </Head>

      <header className="bg-white shadow">
        <div className="max-w-7xl mx-auto py-6 px-4 sm:px-6 lg:px-8">
          <h1 className="text-3xl font-bold text-gray-900">ColdEmailIO</h1>
        </div>
      </header>

      <main>
        <div className="max-w-7xl mx-auto py-6 sm:px-6 lg:px-8">
          <div className="px-4 py-6 sm:px-0">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
              {/* Form Section */}
              <div className="bg-white overflow-hidden shadow rounded-lg p-6">
                <h2 className="text-lg font-medium text-gray-900 mb-4">Generate Personalized Email</h2>
                
                <form onSubmit={handleSubmit(onSubmit)} className="space-y-6">
                  <div className="bg-gray-50 p-4 rounded-md">
                    <h3 className="text-md font-medium text-gray-700 mb-3">Recipient Information</h3>
                    <div className="grid grid-cols-1 gap-4">
                      <div>
                        <label htmlFor="name" className="block text-sm font-medium text-gray-700">
                          Name*
                        </label>
                        <input
                          type="text"
                          id="name"
                          className="mt-1 focus:ring-blue-500 focus:border-blue-500 block w-full shadow-sm sm:text-sm border-gray-300 rounded-md"
                          {...register('recipient.name', { required: 'Name is required' })}
                        />
                        {errors.recipient?.name && (
                          <p className="mt-1 text-sm text-red-600">{errors.recipient.name.message}</p>
                        )}
                      </div>
                      
                      <div>
                        <label htmlFor="email" className="block text-sm font-medium text-gray-700">
                          Email*
                        </label>
                        <input
                          type="email"
                          id="email"
                          className="mt-1 focus:ring-blue-500 focus:border-blue-500 block w-full shadow-sm sm:text-sm border-gray-300 rounded-md"
                          {...register('recipient.email', { 
                            required: 'Email is required',
                            pattern: {
                              value: /^[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}$/i,
                              message: 'Invalid email address'
                            }
                          })}
                        />
                        {errors.recipient?.email && (
                          <p className="mt-1 text-sm text-red-600">{errors.recipient.email.message}</p>
                        )}
                      </div>
                      
                      <div>
                        <label htmlFor="company" className="block text-sm font-medium text-gray-700">
                          Company
                        </label>
                        <input
                          type="text"
                          id="company"
                          className="mt-1 focus:ring-blue-500 focus:border-blue-500 block w-full shadow-sm sm:text-sm border-gray-300 rounded-md"
                          {...register('recipient.company')}
                        />
                      </div>
                      
                      <div>
                        <label htmlFor="role" className="block text-sm font-medium text-gray-700">
                          Role
                        </label>
                        <input
                          type="text"
                          id="role"
                          className="mt-1 focus:ring-blue-500 focus:border-blue-500 block w-full shadow-sm sm:text-sm border-gray-300 rounded-md"
                          {...register('recipient.role')}
                        />
                      </div>
                      
                      <div>
                        <label htmlFor="linkedin_url" className="block text-sm font-medium text-gray-700">
                          LinkedIn URL
                        </label>
                        <input
                          type="url"
                          id="linkedin_url"
                          className="mt-1 focus:ring-blue-500 focus:border-blue-500 block w-full shadow-sm sm:text-sm border-gray-300 rounded-md"
                          {...register('recipient.linkedin_url')}
                        />
                      </div>
                    </div>
                  </div>

                  <div>
                    <label htmlFor="goal" className="block text-sm font-medium text-gray-700">
                      Your Goal*
                    </label>
                    <textarea
                      id="goal"
                      rows={3}
                      className="mt-1 focus:ring-blue-500 focus:border-blue-500 block w-full shadow-sm sm:text-sm border-gray-300 rounded-md"
                      placeholder="e.g., Request a coffee chat to discuss AI strategy"
                      {...register('goal', { required: 'Goal is required' })}
                    ></textarea>
                    {errors.goal && (
                      <p className="mt-1 text-sm text-red-600">{errors.goal.message}</p>
                    )}
                  </div>

                  <div>
                    <label htmlFor="additional_context" className="block text-sm font-medium text-gray-700">
                      Additional Context
                    </label>
                    <textarea
                      id="additional_context"
                      rows={3}
                      className="mt-1 focus:ring-blue-500 focus:border-blue-500 block w-full shadow-sm sm:text-sm border-gray-300 rounded-md"
                      placeholder="Any specific information you'd like to include"
                      {...register('additional_context')}
                    ></textarea>
                  </div>

                  <div>
                    <label htmlFor="tone" className="block text-sm font-medium text-gray-700">
                      Email Tone
                    </label>
                    <select
                      id="tone"
                      className="mt-1 focus:ring-blue-500 focus:border-blue-500 block w-full shadow-sm sm:text-sm border-gray-300 rounded-md"
                      {...register('tone')}
                    >
                      <option value="professional">Professional</option>
                      <option value="friendly">Friendly</option>
                      <option value="casual">Casual</option>
                      <option value="formal">Formal</option>
                    </select>
                  </div>

                  <div className="flex items-center">
                    <input
                      id="use_lead_scoring"
                      type="checkbox"
                      className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
                      {...register('use_lead_scoring')}
                    />
                    <label htmlFor="use_lead_scoring" className="ml-2 block text-sm text-gray-700">
                      Use lead scoring to optimize approach
                    </label>
                  </div>

                  <div>
                    <button
                      type="submit"
                      disabled={isLoading}
                      className="w-full flex justify-center py-2 px-4 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 disabled:opacity-50"
                    >
                      {isLoading ? 'Generating...' : 'Generate Email Draft'}
                    </button>
                  </div>
                </form>
                
                {error && (
                  <div className="mt-4 p-3 bg-red-50 rounded-md">
                    <p className="text-sm text-red-600">{error}</p>
                  </div>
                )}
              </div>

              {/* Result Section */}
              <div className="bg-white overflow-hidden shadow rounded-lg p-6">
                <h2 className="text-lg font-medium text-gray-900 mb-4">Generated Email</h2>
                
                {!emailDraft ? (
                  <div className="text-center py-12">
                    <p className="text-gray-500">
                      Fill out the form and generate an email draft to see the result here.
                    </p>
                  </div>
                ) : (
                  <div className="space-y-6">
                    {emailDraft.lead_score !== undefined && (
                      <div className="bg-blue-50 p-4 rounded-md">
                        <div className="flex items-center">
                          <div className="text-blue-400">
                            <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6" />
                            </svg>
                          </div>
                          <div className="ml-3">
                            <h3 className="text-sm font-medium text-blue-800">Lead Score</h3>
                            <div className="mt-1 text-sm text-blue-700">
                              <p>Score: {Math.round(emailDraft.lead_score * 100)}%</p>
                              <p className="text-xs mt-1">
                                {emailDraft.lead_score >= 0.8 
                                  ? 'High-value lead! Personalized follow-up recommended.' 
                                  : emailDraft.lead_score >= 0.6 
                                  ? 'Promising lead. Highlight relevant use cases.'
                                  : 'Nurture this lead with educational content.'}
                              </p>
                            </div>
                          </div>
                        </div>
                      </div>
                    )}

                    <div>
                      <h3 className="text-md font-medium text-gray-700">Subject</h3>
                      <p className="mt-1 text-sm text-gray-900">{emailDraft.subject}</p>
                    </div>

                    <div>
                      <h3 className="text-md font-medium text-gray-700">Body</h3>
                      <div className="mt-1 p-4 border border-gray-200 rounded-md bg-gray-50">
                        <pre className="text-sm text-gray-900 whitespace-pre-wrap font-sans">{emailDraft.body}</pre>
                      </div>
                    </div>

                    <div className="flex space-x-4">
                      <button
                        type="button"
                        className="flex-1 py-2 px-4 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500"
                        onClick={() => {
                          navigator.clipboard.writeText(`Subject: ${emailDraft.subject}\n\n${emailDraft.body}`);
                          alert('Email copied to clipboard!');
                        }}
                      >
                        Copy to Clipboard
                      </button>
                      
                      <button
                        type="button"
                        className="flex-1 py-2 px-4 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-green-600 hover:bg-green-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-green-500"
                        onClick={handlePushToHubspot}
                      >
                        Push to HubSpot
                      </button>
                    </div>
                    
                    {pushStatus === 'success' && (
                      <div className="p-3 bg-green-50 rounded-md">
                        <p className="text-sm text-green-600">Successfully pushed to HubSpot!</p>
                      </div>
                    )}
                    
                    {pushStatus === 'error' && (
                      <div className="p-3 bg-red-50 rounded-md">
                        <p className="text-sm text-red-600">Failed to push to HubSpot. Please try again.</p>
                      </div>
                    )}
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      </main>

      <footer className="bg-white shadow-inner">
        <div className="max-w-7xl mx-auto py-4 px-4 sm:px-6 lg:px-8">
          <p className="text-center text-sm text-gray-500">
            ColdEmailIO - AI-powered cold email personalization
          </p>
        </div>
      </footer>
    </div>
  )
}
