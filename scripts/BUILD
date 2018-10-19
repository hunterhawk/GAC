package(default_visibility = ["//visibility:public"])

py_binary(
    name = "train",
    srcs = ["train.py"],
    deps = ["//gac:lib"]
)

py_binary(
    name = "eval",
    srcs = ["eval.py"],
    deps = ["//gac:lib"]
)

py_binary(
    name = "build_tfrecords",
    srcs = ["build_tfrecords.py"],
    deps = ["//gac:lib"],
)

py_binary(
    name = "saved_model",
    srcs = ["saved_model.py"],
    deps = ["//gac:lib"]
)