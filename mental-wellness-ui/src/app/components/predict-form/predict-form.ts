import { Component, EventEmitter, Output } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormBuilder, FormGroup, ReactiveFormsModule, Validators } from '@angular/forms';
import { Predict } from '../../services/predict';

@Component({
  selector: 'app-predict-form',
  standalone: true,
  imports: [CommonModule, ReactiveFormsModule],
  templateUrl: './predict-form.html',
  styleUrls: ['./predict-form.scss'],
})
export class PredictForm {
  @Output() predictionMade: EventEmitter<number | null> = new EventEmitter<number | null>();

  wellnessForm: FormGroup;

  occupations = ['Employed', 'Student', 'Self-employed', 'Unemployed', 'Retired'];
  genders = ['Male', 'Female'];
  workModes = ['Remote', 'In-person', 'Hybrid'];

  result: number | null = null;
  loading = false;
  error: string | null = null;

  constructor(private fb: FormBuilder, private predictService: Predict) {
    this.wellnessForm = this.fb.group({
      age: [30, Validators.required], // default: 30 years old
      gender: ['Male', Validators.required],
      occupation: ['Employed', Validators.required],
      work_mode: ['Remote', Validators.required],

      screen_time_hours: [6, Validators.required],
      work_screen_hours: [4, Validators.required],
      leisure_screen_hours: [2, Validators.required],

      sleep_hours: [7, Validators.required],
      sleep_quality_1_5: [4, [Validators.required, Validators.min(1), Validators.max(5)]],

      stress_level_0_10: [5, [Validators.required, Validators.min(0), Validators.max(10)]],
      productivity_0_100: [80, [Validators.required, Validators.min(0), Validators.max(100)]],

      exercise_minutes_per_week: [150, Validators.required],
      social_hours_per_week: [10, Validators.required],
    });
  }

  submit() {
    if (this.wellnessForm.invalid) return;

    this.loading = true;
    this.error = null;
    this.result = null;

    const payload = this.wellnessForm.value;

    this.predictService.predict(payload).subscribe({
      next: (res) => {
        this.result = res.mental_wellness_index;
        this.predictionMade.emit(this.result); // <-- emit result to parent
        this.loading = false;
      },
      error: () => {
        this.error = 'Failed to get prediction';
        this.loading = false;
      },
    });
  }
}
