import os
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader

PAGE_WIDTH, PAGE_HEIGHT = A4


def pt_to_img(pt_name):
    # strip _aug suffix before swapping extension, to handle augmented filenames
    name = pt_name.replace("_aug.pt", ".pt")
    return os.path.splitext(name)[0] + ".jpg"


def create_retrieval_pdf(
    output_path,
    query_item_id,
    retrieved_item_ids,
    images_dir,
    distances=None,
    retrieved_metadata=None,
    query_metadata=None,
    result_dir="results",
):
    output_path_folder = os.path.join(result_dir, output_path)

    c = canvas.Canvas(output_path_folder, pagesize=A4)
    

    y = PAGE_HEIGHT - 50

    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, y, "Query Image")
    y -= 30

    query_img = os.path.join(images_dir, pt_to_img(query_item_id))
    c.drawImage(
        ImageReader(query_img),
        50,
        y - 200,
        width=200,
        height=200,
        preserveAspectRatio=True,
        anchor='sw',
    )

    if query_metadata is not None:
        text_x = 270
        text_y = y - 20

        c.setFont("Helvetica", 10)
        c.drawString(text_x, text_y, f"ID: {query_item_id.replace('.pt','')[:40]}")

        c.setFont("Helvetica", 9)
        c.drawString(text_x, text_y - 15, f"Cat: {query_metadata.get('category', '-')}")
        c.drawString(text_x, text_y - 30, f"Type: {query_metadata.get('type', '-')}")

        colors = query_metadata.get("colors", [])
        if colors:
            c.drawString(text_x, text_y - 45, f"Colors: {', '.join(colors[:4])}")

        season = query_metadata.get("season")
        if season:
            c.drawString(text_x, text_y - 60, f"Season: {season}")

    y -= 240
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, y, "Retrieved Results")
    y -= 30

    c.setFont("Helvetica", 10)

    x = 50
    img_size = 120
    gap = 20
    card_height = img_size + 90

    for i, item_id in enumerate(retrieved_item_ids):
        img_path = os.path.join(images_dir, pt_to_img(item_id))

        if not os.path.exists(img_path):
            continue

        c.drawImage(
            ImageReader(img_path),
            x,
            y - img_size,
            width=img_size,
            height=img_size,
            preserveAspectRatio=True,
            anchor='sw'
        )


        text_y = y - img_size - 12
        c.setFont("Helvetica", 8)

        c.drawString(
            x,
            text_y,
            f"ID: {item_id.replace('.pt','')[:28]}"
        )

        if distances is not None:
            c.drawString(
                x,
                text_y - 10,
                f"Dist: {distances[i]:.3f}"
            )

        if retrieved_metadata is not None:
            meta = retrieved_metadata[i]

            c.drawString(
                x,
                text_y - 20,
                f"Cat: {meta.get('category', '-')}"
            )

            c.drawString(
                x,
                text_y - 30,
                f"Type: {meta.get('type', '-')}"
            )

            colors = meta.get("colors", [])
            if colors:
                c.drawString(
                    x,
                    text_y - 40,
                    f"Colors: {', '.join(colors[:3])}"
                )

            season = meta.get("season")
            if season:
                c.drawString(
                    x,
                    text_y - 50,
                    f"Season: {season}"
                )


        x += img_size + gap

        if x + img_size > PAGE_WIDTH:
            x = 50
            y -= card_height

            if y < card_height + 50:
                c.showPage()
                y = PAGE_HEIGHT - 50


    c.save()