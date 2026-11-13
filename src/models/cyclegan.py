"""Composición CycleGAN y un paso real de optimización de sus cuatro modelos."""

import math

import tensorflow as tf

from src.models.generator import build_generator
from src.models.discriminator import build_discriminator


class CycleGAN(tf.Module):
    """Integra modelos exportables por separado y optimizadores independientes.

    Cada generador minimiza adversaria + lambda_cycle * (ciclo_A + ciclo_B)
    + lambda_identity * identidad de su dominio de salida. Las métricas de ciclo
    e identidad son medias L1 sin ponderar; las de generadores incluyen los pesos.
    """

    def __init__(self, config: dict):
        super().__init__(name="CycleGAN")
        self.input_shape = tuple(config["model"]["input_shape"])
        training = config["training"]
        self.lambda_cycle = float(training["lambda_cycle"])
        self.lambda_identity = float(training.get("lambda_identity", 0.0))
        learning_rate = float(training["learning_rate"])
        beta_1 = float(training.get("beta_1", 0.5))
        beta_2 = float(training.get("beta_2", 0.999))
        if not math.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("learning_rate debe ser finito y positivo")
        if any(not math.isfinite(value) or value < 0
               for value in (self.lambda_cycle, self.lambda_identity)):
            raise ValueError("lambda_cycle y lambda_identity deben ser finitos y no negativos")
        if not (0 <= beta_1 < 1 and 0 <= beta_2 < 1):
            raise ValueError("beta_1 y beta_2 deben pertenecer a [0, 1)")

        self.G_A2B = build_generator(self.input_shape, name="G_A2B")
        self.G_B2A = build_generator(self.input_shape, name="G_B2A")
        self.D_A = build_discriminator(self.input_shape, name="D_A")
        self.D_B = build_discriminator(self.input_shape, name="D_B")
        self.optimizers = {
            name: tf.keras.optimizers.Adam(
                learning_rate=learning_rate, beta_1=beta_1, beta_2=beta_2,
                name=f"Adam_{name}",
            )
            for name in ("G_A2B", "G_B2A", "D_A", "D_B")
        }
        self.adversarial_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    @staticmethod
    def l1_loss(target, prediction):
        """Media absoluta sobre batch, tiempo, frecuencia y canal."""
        return tf.reduce_mean(tf.abs(target - prediction))

    def generator_loss(self, fake_logits):
        """El generador intenta que los parches generados se clasifiquen reales."""
        return self.adversarial_loss(tf.ones_like(fake_logits), fake_logits)

    def discriminator_loss(self, real_logits, fake_logits):
        """Media de BCE real y falsa; ambas entradas son logits."""
        real = self.adversarial_loss(tf.ones_like(real_logits), real_logits)
        fake = self.adversarial_loss(tf.zeros_like(fake_logits), fake_logits)
        return 0.5 * (real + fake)

    def train_step(self, batch_A, batch_B):
        """Actualiza los cuatro modelos y devuelve métricas escalares TensorFlow.

        A y B son batches float32 independientes en [-1, 1]. No se cambia su
        normalización. El método puede envolverse en tf.function.
        """
        batch_A = tf.ensure_shape(tf.convert_to_tensor(batch_A, dtype=tf.float32),
                                  (None, *self.input_shape))
        batch_B = tf.ensure_shape(tf.convert_to_tensor(batch_B, dtype=tf.float32),
                                  (None, *self.input_shape))
        with tf.GradientTape(persistent=True) as tape:
            fake_B = self.G_A2B(batch_A, training=True)
            fake_A = self.G_B2A(batch_B, training=True)
            cycled_A = self.G_B2A(fake_B, training=True)
            cycled_B = self.G_A2B(fake_A, training=True)
            real_A_logits = self.D_A(batch_A, training=True)
            real_B_logits = self.D_B(batch_B, training=True)
            fake_A_logits = self.D_A(fake_A, training=True)
            fake_B_logits = self.D_B(fake_B, training=True)

            cycle_A = self.l1_loss(batch_A, cycled_A)
            cycle_B = self.l1_loss(batch_B, cycled_B)
            cycle = cycle_A + cycle_B
            adversarial_A2B = self.generator_loss(fake_B_logits)
            adversarial_B2A = self.generator_loss(fake_A_logits)
            loss_A2B = adversarial_A2B + self.lambda_cycle * cycle
            loss_B2A = adversarial_B2A + self.lambda_cycle * cycle
            metrics = {
                "G_A2B_adversarial": adversarial_A2B,
                "G_B2A_adversarial": adversarial_B2A,
                "D_A_loss": self.discriminator_loss(real_A_logits, fake_A_logits),
                "D_B_loss": self.discriminator_loss(real_B_logits, fake_B_logits),
                "cycle_A": cycle_A,
                "cycle_B": cycle_B,
                "cycle_loss": cycle,
            }
            if self.lambda_identity > 0:
                identity_A = self.l1_loss(batch_A, self.G_B2A(batch_A, training=True))
                identity_B = self.l1_loss(batch_B, self.G_A2B(batch_B, training=True))
                loss_A2B += self.lambda_identity * identity_B
                loss_B2A += self.lambda_identity * identity_A
                metrics.update(identity_A=identity_A, identity_B=identity_B,
                               identity_loss=identity_A + identity_B)
            metrics.update(G_A2B_loss=loss_A2B, G_B2A_loss=loss_B2A)

        updates = []
        for name, optimizer in self.optimizers.items():
            variables = getattr(self, name).trainable_variables
            gradients = tape.gradient(metrics[f"{name}_loss"], variables)
            for gradient in gradients:
                if gradient is None:
                    raise RuntimeError(f"Gradiente desconectado en {name}")
                tf.debugging.assert_all_finite(gradient, f"Gradiente no finito en {name}")
            updates.append((optimizer, gradients, variables))
        del tape
        for name, value in metrics.items():
            tf.debugging.assert_all_finite(value, f"Métrica no finita: {name}")
        for optimizer, gradients, variables in updates:
            optimizer.apply_gradients(zip(gradients, variables))
        return metrics


def build_cyclegan(config: dict) -> CycleGAN:
    """Crea el integrador con las arquitecturas e hiperparámetros del proyecto."""
    return CycleGAN(config)
