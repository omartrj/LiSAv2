import numpy as np

class KalmanTracker:
    def __init__(self, dt=0.05, process_noise=2.0, measurement_noise=0.5):
        """
        Filtro di Kalman Lineare in 2D (Modello a Velocità Costante).
        Traccia (x, y) e le loro velocità (vx, vy).
        """
        self.dt = dt
        # Stato: [x, y, vx, vy]^T
        self.state = np.zeros((4, 1))
        # Covarianza dell'errore
        self.P = np.eye(4) * 10.0
        
        # Matrice di transizione di stato (F)
        self.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        
        # Matrice di misurazione (H) - Osserviamo solo [x, y]
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])
        
        # Rumore di processo (Q) - Incertezza sull'accelerazione
        q = process_noise
        self.Q = np.array([
            [0.25*dt**4, 0,          0.5*dt**3, 0],
            [0,          0.25*dt**4, 0,         0.5*dt**3],
            [0.5*dt**3,  0,          dt**2,     0],
            [0,          0.5*dt**3,  0,         dt**2]
        ]) * q
        
        # Rumore di misura (R) - Incertezza del modello Neurale
        r = measurement_noise
        self.R = np.eye(2) * r
        
        self.initialized = False

    def reset(self):
        self.initialized = False
        self.state = np.zeros((4, 1))
        self.P = np.eye(4) * 10.0

    def update(self, z_x, z_y, is_active=True):
        if not is_active:
            # Se la sirena non c'è, facciamo solo la Previsione Cieca (coasting)
            if self.initialized:
                self.state = self.F @ self.state
                self.P = self.F @ self.P @ self.F.T + self.Q
                return self.state[0, 0], self.state[1, 0]
            else:
                return 0.0, 0.0 # Restiamo fermi se non siamo mai partiti
                
        z = np.array([[z_x], [z_y]])
        
        if not self.initialized:
            # Primo frame in assoluto, inizializza stato a questa posizione, velocità 0
            self.state[0, 0] = z_x
            self.state[1, 0] = z_y
            self.initialized = True
            return z_x, z_y
            
        # 1. PREDICT (Guarda la dinamica passata)
        state_pred = self.F @ self.state
        P_pred = self.F @ self.P @ self.F.T + self.Q
        
        # 2. UPDATE (Corregge la predizione con la nuova misurazione Z che è ATTIVA)
        y = z - (self.H @ state_pred) # Residuo
        S = self.H @ P_pred @ self.H.T + self.R
        K = P_pred @ self.H.T @ np.linalg.inv(S) # Guadagno di Kalman
        
        self.state = state_pred + (K @ y)
        self.P = (np.eye(4) - K @ self.H) @ P_pred
        
        # Ritorna l'estimate aggiornato (x, y)
        return self.state[0, 0], self.state[1, 0]


class PostProcessor:
    def __init__(self):
        """
        Gestisce lo smoothing temporale delle predizioni utilizzando un Filtro di Kalman Lineare in 2D.
        """
        self.kalman = KalmanTracker(dt=0.05, process_noise=5.0, measurement_noise=1.5)
        self.inactive_frames_count = 0
        self.inactive_threshold = 10 # Dopo mezzo secondo (10 frame * 0.05s) di inattività, resetti
        
    def reset(self):
        """Pulisce la memoria (utile quando cambia la sequenza)."""
        self.kalman.reset()
        self.inactive_frames_count = 0
        
    def update(self, raw_dist, raw_angle, is_active=True):
        """
        Aggiunge una nuova predizione grezza e restituisce quella filtrata in tempo reale.
        Converte prima in Coordinate Cartesiane interne per preservare la linearità del modello.
        """
        if not is_active:
            self.inactive_frames_count += 1
            if self.inactive_frames_count >= self.inactive_threshold:
                self.kalman.reset() # Se inattiva da troppo, perdi la traccia
        else:
            self.inactive_frames_count = 0 # Sirena tornata attiva
            
        # 1. Converti Misura Polare in Cartesiana
        rad = np.deg2rad(raw_angle)
        x_meas = raw_dist * np.cos(rad)
        y_meas = raw_dist * np.sin(rad)
        
        # 2. Update Ricorsivo del Filtro di Kalman passando l'attività
        x_est, y_est = self.kalman.update(x_meas, y_meas, is_active=is_active)
        
        # 3. Riconverti Stima Cartesiana in Polare
        smooth_dist = np.sqrt(x_est**2 + y_est**2)
        smooth_angle_rad = np.arctan2(y_est, x_est)
        smooth_angle = ((np.rad2deg(smooth_angle_rad) + 180) % 360) - 180
        
        # Forziamo a zero l'uscita solo visivamente se è inattiva, 
        # ma Kalman sta tenendo memoria del Predict "fantasma" finché non si resetta.
        if not is_active and not self.kalman.initialized:
            smooth_dist = 0.0
            smooth_angle = 0.0
            
        return smooth_dist, smooth_angle