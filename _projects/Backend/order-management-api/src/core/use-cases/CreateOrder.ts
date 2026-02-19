import { Order, OrderStatus } from '../domain/Order';
import { IOrderRepository } from '../repositories/IOrderRepository';

// Estructura de datos que esperamos de la API (DTO simple)
interface CreateOrderRequest {
  customerName: string;
  items: string[];
  totalAmount: number;
}

export class CreateOrder {
  // Recibimos la INTERFAZ, no la implementación real (Inyección de dependencias)
  constructor(private orderRepository: IOrderRepository) {}

  async execute(request: CreateOrderRequest): Promise<Order> {
    const newOrder = new Order({
      id: Math.random().toString(36).substring(7), // Temporal, luego usaremos UUID
      customerName: request.customerName,
      items: request.items,
      totalAmount: request.totalAmount,
      status: 'PENDING',
      createdAt: new Date()
    });

    await this.orderRepository.save(newOrder);
    return newOrder;
  }
}