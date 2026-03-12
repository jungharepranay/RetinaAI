"""
model.py
--------
DenseNet121-based multi-label classifier for retinal disease detection.

Supports both BinaryCrossentropy and Focal Loss.
"""

import tensorflow as tf
from tensorflow.keras import layers, Model


NUM_CLASSES = 8  # N, D, G, C, A, H, M, O


# -------------------- Focal Loss -------------------- #

def focal_loss(gamma=2.0, alpha=0.25):
    """
    Focal Loss for multi-label classification.

    Focal loss down-weights well-classified examples and focuses
    training on hard, misclassified examples — especially useful
    for imbalanced datasets like ODIR-5K.

    Parameters
    ----------
    gamma : float
        Focusing parameter. Higher values increase focus on hard examples.
        ``gamma=0`` reduces to standard binary cross-entropy.
    alpha : float
        Balancing factor for positive vs negative classes.

    Returns
    -------
    loss_fn : callable
        A loss function compatible with ``model.compile(loss=...)``.

    References
    ----------
    Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017.
    """
    def _focal_loss_fn(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        # Clip predictions to prevent log(0)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)

        # Binary cross-entropy per element
        bce = -(y_true * tf.math.log(y_pred) +
                (1.0 - y_true) * tf.math.log(1.0 - y_pred))

        # Focal modulating factor
        p_t = y_true * y_pred + (1.0 - y_true) * (1.0 - y_pred)
        focal_weight = tf.pow(1.0 - p_t, gamma)

        # Alpha balancing
        alpha_t = y_true * alpha + (1.0 - y_true) * (1.0 - alpha)

        loss = alpha_t * focal_weight * bce
        return tf.reduce_mean(loss)

    return _focal_loss_fn


# -------------------- Model Builder -------------------- #

def _get_loss(loss_type='bce', focal_gamma=2.0, focal_alpha=0.25):
    """Return the appropriate loss function."""
    if loss_type == 'focal':
        return focal_loss(gamma=focal_gamma, alpha=focal_alpha)
    return tf.keras.losses.BinaryCrossentropy()


def _get_metrics():
    """Standard metrics used across compile calls."""
    return [
        tf.keras.metrics.AUC(name='auc', multi_label=True),
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall'),
    ]


def build_model(input_shape=(224, 224, 3), num_classes=NUM_CLASSES,
                learning_rate=1e-4, loss_type='bce'):
    """
    Build and compile a DenseNet121-based multi-label model.

    Architecture
    ------------
    Input → DenseNet121 (ImageNet, no top) → GlobalAveragePooling2D
    → BatchNormalization → Dense(512, relu) → Dropout(0.5)
    → Dense(num_classes, sigmoid)

    Parameters
    ----------
    input_shape : tuple
    num_classes : int
    learning_rate : float
    loss_type : str
        ``'bce'`` for BinaryCrossentropy (default),
        ``'focal'`` for Focal Loss.

    Returns
    -------
    tf.keras.Model  (compiled)
    """
    base = tf.keras.applications.DenseNet121(
        include_top=False,
        weights='imagenet',
        input_shape=input_shape,
    )
    base.trainable = False  # freeze for Phase 1

    inputs = layers.Input(shape=input_shape)
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='sigmoid')(x)

    model = Model(inputs, outputs, name='retinal_disease_classifier')

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=_get_loss(loss_type),
        metrics=_get_metrics(),
    )
    return model


def unfreeze_top_layers(model, num_layers_to_unfreeze=50,
                        learning_rate=1e-5, loss_type='bce'):
    """
    Unfreeze the top *num_layers_to_unfreeze* layers of the DenseNet
    backbone and re-compile with a lower learning rate (Phase 2).
    """
    base = model.layers[1]  # DenseNet121
    base.trainable = True

    # Freeze all layers except the last N
    for layer in base.layers[:-num_layers_to_unfreeze]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=_get_loss(loss_type),
        metrics=_get_metrics(),
    )
    return model


if __name__ == "__main__":
    m = build_model()
    m.summary()
