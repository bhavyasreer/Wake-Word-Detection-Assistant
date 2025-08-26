import os
import pickle
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
import io

# If modifying these scopes, delete the file token.pickle.
SCOPES = ['https://www.googleapis.com/auth/drive.readonly']

def get_drive_service():
    """Get or create Google Drive service."""
    creds = None
    # The file token.pickle stores the user's access and refresh tokens
    if os.path.exists('token.pickle'):
        with open('token.pickle', 'rb') as token:
            creds = pickle.load(token)
    
    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        # Save the credentials for the next run
        with open('token.pickle', 'wb') as token:
            pickle.dump(creds, token)

    return build('drive', 'v3', credentials=creds)

def download_file(service, file_id, output_path):
    """Download a file from Google Drive."""
    request = service.files().get_media(fileId=file_id)
    fh = io.BytesIO()
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while done is False:
        status, done = downloader.next_chunk()
        print(f"Download {int(status.progress() * 100)}%")
    
    # Save the file
    with open(output_path, 'wb') as f:
        f.write(fh.getvalue())

def list_files_in_folder(service, folder_id):
    """List all files in a Google Drive folder."""
    results = service.files().list(
        q=f"'{folder_id}' in parents",
        pageSize=1000,
        fields="nextPageToken, files(id, name)"
    ).execute()
    return results.get('files', [])

def download_folder(service, folder_id, local_path):
    """Download all files from a Google Drive folder."""
    if not os.path.exists(local_path):
        os.makedirs(local_path)
    
    files = list_files_in_folder(service, folder_id)
    for file in files:
        file_path = os.path.join(local_path, file['name'])
        print(f"Downloading {file['name']}...")
        download_file(service, file['id'], file_path)
        print(f"Downloaded {file['name']}")

def setup_drive_dataset(wake_word_folder_id, background_folder_id):
    """Set up the dataset by downloading files from Google Drive."""
    service = get_drive_service()
    
    # Create data directories
    os.makedirs('data/wake_word', exist_ok=True)
    os.makedirs('data/background', exist_ok=True)
    
    # Download wake word samples
    print("Downloading wake word samples...")
    download_folder(service, wake_word_folder_id, 'data/wake_word')
    
    # Download background samples
    print("\nDownloading background samples...")
    download_folder(service, background_folder_id, 'data/background')
    
    print("\nDataset setup complete!") 