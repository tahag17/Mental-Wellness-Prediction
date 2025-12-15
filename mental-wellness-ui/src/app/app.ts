import { Component, signal } from '@angular/core';
import { RouterOutlet } from '@angular/router';
import { PredictForm } from './components/predict-form/predict-form';
import { PredictHeader } from './components/predict-header/predict-header';
import { PredictFooter } from './components/predict-footer/predict-footer';

@Component({
  selector: 'app-root',
  imports: [RouterOutlet, PredictForm, PredictHeader, PredictFooter],
  templateUrl: './app.html',
  styleUrl: './app.scss',
})
export class App {
  protected readonly title = signal('mental-wellness-ui');

  predictionResult: number | null = null;

  onPrediction(result: number | null) {
    this.predictionResult = result;
  }
}
