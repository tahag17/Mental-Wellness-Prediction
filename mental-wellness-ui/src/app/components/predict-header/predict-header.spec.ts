import { ComponentFixture, TestBed } from '@angular/core/testing';

import { PredictHeader } from './predict-header';

describe('PredictHeader', () => {
  let component: PredictHeader;
  let fixture: ComponentFixture<PredictHeader>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [PredictHeader]
    })
    .compileComponents();

    fixture = TestBed.createComponent(PredictHeader);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
