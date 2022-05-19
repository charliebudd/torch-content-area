from torchcontentarea import ContentAreaInference
from testing.utils import TestDataset, perimeter_distance_score
import matplotlib.pyplot as plt

def show(img, gt_area, area, points_x, points_y, norm_x, norm_y):
    norm_y = [-y for y in norm_y]
    plt.rcParams["figure.figsize"] = (15, 10)
    plt.subplot(111).axis("off")
    plt.imshow(img.permute(1, 2, 0).cpu())
    plt.scatter(points_x, points_y, color="green", s=80)
    plt.quiver(points_x, points_y, norm_x, norm_y, color="green")
    if area != None:
        patch = plt.Circle((area[0], area[1]), area[2], color='green', fill=False)
        plt.gca().add_patch(patch)
    if gt_area != None:
        patch = plt.Circle((gt_area[0], gt_area[1]), gt_area[2], color='blue', fill=False)
        plt.gca().add_patch(patch)
    plt.xlim((0, img.shape[2]))
    plt.ylim((img.shape[1], 0))
    plt.show()

content_area = ContentAreaInference()
dataset = TestDataset()

show_bad_circles = True
show_bad_classification = True

area_scores = []
no_area_scores = []

correct_scores = []
incorrect_scores = []

for index, (img, gt_area) in enumerate(dataset):

    img = img.cuda()
    area = content_area.infer_area(img)
    points_x, points_y, norm_x, norm_y, _, confidence_scores = content_area.get_debug(img)
    expect_area = gt_area != None
    area_found = area != None

    if expect_area and area_found:
        score = perimeter_distance_score(gt_area, area)
        if show_bad_circles and score > 10:
            print(f"Bad Circle...")
            print(f"Error: {int(score)}px")
            show(img, gt_area, area, points_x, points_y, norm_x, norm_y)

    if show_bad_classification and expect_area != area_found:
        print(f"False Positive..." if area_found else "False Negative...")
        print(f"Border Score: {confidence_scores[0]:.3f}")
        print(f"Circle Score: {confidence_scores[1]:.3f}")
        print(f"Final Score: {confidence_scores[2]:.3f}")
        show(img, gt_area, area, points_x, points_y, norm_x, norm_y)

    if expect_area:
        area_scores.append(confidence_scores)
    else:
        no_area_scores.append(confidence_scores)
        
    if expect_area == area_found:
        correct_scores.append(confidence_scores)
    else:
        incorrect_scores.append(confidence_scores)


area_border_scores, area_circle_scores, area_final_scores = zip(*area_scores)
no_area_border_scores, no_area_circle_scores, no_area_final_scores = zip(*no_area_scores)

plt.rcParams["figure.figsize"] = (15, 5)
plt.subplot(131)
plt.title("Border Score")
plt.ylabel("Sample Count")
plt.xlabel("Score (0-1)")
plt.hist([area_border_scores, no_area_border_scores], bins=20)
plt.xlim(0, 1)
plt.subplot(132)
plt.title("Circle Score")
plt.xlabel("Score (0-1)")
plt.hist([area_circle_scores, no_area_circle_scores], bins=20)
plt.xlim(0, 1)
plt.subplot(133)
plt.title("Final Score")
plt.xlabel("Score (0-1)")
plt.hist([area_final_scores, no_area_final_scores], bins=20)
plt.plot([0.15, 0.15], [0, 1400], color="red")
plt.xlim(0, 1)
plt.show()


correct_border_scores, correct_circle_scores, correct_final_scores = zip(*correct_scores)
incorrect_border_scores, incorrect_circle_scores, incorrect_final_scores = zip(*incorrect_scores)

plt.rcParams["figure.figsize"] = (15, 5)
plt.subplot(131)
plt.title("Border Score")
plt.ylabel("Sample Count")
plt.xlabel("Score (0-1)")
plt.hist([incorrect_border_scores], bins=20)
plt.xlim(0, 1)
plt.subplot(132)
plt.title("Circle Score")
plt.xlabel("Score (0-1)")
plt.hist([incorrect_circle_scores], bins=20)
plt.xlim(0, 1)
plt.subplot(133)
plt.title("Final Score")
plt.xlabel("Score (0-1)")
plt.hist([incorrect_final_scores], bins=20)
plt.plot([0.15, 0.15], [0, 1400], color="red")
plt.xlim(0, 1)
plt.show()
