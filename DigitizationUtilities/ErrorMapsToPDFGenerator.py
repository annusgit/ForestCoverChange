from fpdf import FPDF
import os

rgb_path = 'E:\Forest Cover - Redo 2020\Trainings and Results\Error Maps\model_48_topologyENC_4_DEC_4_lr1e-06_bands3'
full_spectrum_path = 'E:\Forest Cover - Redo 2020\Trainings and Results\Error Maps\model_14_topologyENC_4_DEC_4_lr1e-06_bands11'
augmented_path = 'E:\Forest Cover - Redo 2020\Trainings and Results\Error Maps\model_53_topologyENC_4_DEC_4_lr1e-06_bands18'
indices_path = 'E:\Forest Cover - Redo 2020\Trainings and Results\Error Maps\model_39_topologyENC_4_DEC_4_lr1e-06_bands7'
destination_path = 'E:\Forest Cover - Redo 2020\Trainings and Results\Error Maps'


def main():
    pdf = FPDF()
    default_A4_pdf_width = 210
    default_A4_pdf_height = 297
    all_test_samples = os.listdir(rgb_path)
    page_number = 1
    for idx, test_sample in enumerate(all_test_samples):
        print("[LOG] Adding {} ({}/{})".format(test_sample, idx+1, len(all_test_samples)))
        rgb_test_sample = os.path.join(rgb_path, test_sample)
        full_spectrum_test_sample = os.path.join(full_spectrum_path, test_sample)
        augmented_test_sample = os.path.join(augmented_path, test_sample)
        indices_test_sample = os.path.join(indices_path, test_sample)
        pdf.add_page()
        pdf.set_font('Arial', 'B', 16)
        pdf.set_text_color(220, 50, 50)
        pdf.set_xy(x=0.0, y=5.0)
        pdf.cell(w=210.0, h=40.0, align='C', txt="[RGB] {}".format(test_sample.split('.')[0]), border=0)
        pdf.image(rgb_test_sample, x=1.0, y=30.0, w=0.95*default_A4_pdf_width, h=0.16*default_A4_pdf_height)
        pdf.set_xy(x=0.0, y=65.0)
        pdf.cell(w=210.0, h=40.0, align='C', txt="[Full-Spectrum] {}".format(test_sample.split('.')[0]), border=0)
        pdf.image(full_spectrum_test_sample, x=1.0, y=90.0, w=0.95*default_A4_pdf_width, h=0.16*default_A4_pdf_height)
        pdf.set_xy(x=0.0, y=125.0)
        pdf.cell(w=210.0, h=40.0, align='C', txt="[Augmented] {}".format(test_sample.split('.')[0]), border=0)
        pdf.image(augmented_test_sample, x=1.0, y=150.0, w=0.95*default_A4_pdf_width, h=0.16*default_A4_pdf_height)
        pdf.set_xy(x=0.0, y=185.0)
        pdf.cell(w=210.0, h=40.0, align='C', txt="[Indices] {}".format(test_sample.split('.')[0]), border=0)
        pdf.image(indices_test_sample, x=1.0, y=210.0, w=0.95*default_A4_pdf_width, h=0.16*default_A4_pdf_height)
        pdf.set_y(266)
        pdf.set_font('Arial', 'I', 8)
        pdf.cell(0, 10, '{}'.format(page_number), 0, 0, 'C')
        page_number += 1
    pdf.output(os.path.join(destination_path, 'ErrorMaps.pdf'), 'F')
    pass


if __name__ == "__main__":
    main()
