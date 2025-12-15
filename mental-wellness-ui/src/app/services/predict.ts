import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';

@Injectable({
  providedIn: 'root',
})
export class Predict {
  private apiUrl =
    // 'https://mental-wellness-prediction-using-artificial-neur-production.up.railway.app/predict';
    'https://projet.fly.dev/predict';

  constructor(private http: HttpClient) {}

  predict(userData: any): Observable<any> {
    return this.http.post<any>(this.apiUrl, userData);
  }
}
