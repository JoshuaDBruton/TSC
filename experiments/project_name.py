# landcover: 4, COCO: 92, IILD: 2, IRCAD: 2, VOC: 22, carla: 13, liver: 3, covid: 4
class ProjectName:
    project_name = ""
    dataset_name = "covid"
    experiment_prefix = "1-covid"
    num_classes = 4
    epochs = 50
    depth = 5
    start_filts = 16
    batch_size = 4
    concat_coords = True
    run_test_loop = dataset_name == "VOC"
    # run_test_loop = False
