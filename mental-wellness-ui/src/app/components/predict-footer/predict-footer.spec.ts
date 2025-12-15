import { ComponentFixture, TestBed } from '@angular/core/testing';

import { PredictFooter } from './predict-footer';

describe('PredictFooter', () => {
  let component: PredictFooter;
  let fixture: ComponentFixture<PredictFooter>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [PredictFooter]
    })
    .compileComponents();

    fixture = TestBed.createComponent(PredictFooter);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
