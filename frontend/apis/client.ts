import { isSSR } from '@dwarvesf/react-utils'

type Headers = Record<string, string>

export class Client {
  headers: Headers = {
    'Content-Type': 'application/json',
  }

  privateHeaders: Headers = {
    ...this.headers,
    credentials: 'include',
  }

  baseUrl: string = process.env.NEXT_PUBLIC_BASE_URL || ''

  public get formDataHeaders(): Headers {
    const cloned = Object.assign({}, this.privateHeaders)
    // Browsers will auto-set Content-Type and other things when formData is used
    // Content-Type must not be present for form data to work
    delete cloned['Content-Type']

    return cloned
  }
}

const client = new Client()

export { client }
