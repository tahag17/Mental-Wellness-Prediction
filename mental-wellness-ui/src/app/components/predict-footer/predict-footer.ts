import { CommonModule } from '@angular/common';
import { Component, Input } from '@angular/core';

@Component({
  selector: 'app-predict-footer',
  standalone: true,
  imports: [CommonModule], // make sure NgIf is included via CommonModule
  templateUrl: './predict-footer.html',
  styleUrls: ['./predict-footer.scss'],
})
export class PredictFooter {
  @Input() result: number | null = null;

  // Interpretation ranges
  get interpretation(): string {
    if (this.result === null) return '';
    if (this.result <= 50)
      return 'Bien‑être faible (stress élevé / risque de détresse psychologique)';
    if (this.result <= 75) return 'Bien‑être modéré (fonctionnement raisonnable)';
    return 'Bien‑être élevé (santé mentale positive)';
  }

  // Optional CSS class for color-coding
  get colorClass(): string {
    if (this.result === null) return '';
    if (this.result <= 50) return 'low';
    if (this.result <= 75) return 'medium';
    return 'high';
  }
}
